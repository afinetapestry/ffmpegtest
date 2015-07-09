//
//  main.cpp
//  ffmpegtest
//
//  Created by Jonathan Hatchett on 6/11/15.
//  Copyright (c) 2015 Jonathan Hatchett. All rights reserved.
//

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/time.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

#include <SDL.h>

#define SDL_AUDIO_BUFFER_SIZE 1024 // In samples

AVDictionary * _options = nullptr;

AVFormatContext * _format = nullptr;

bool _running = true;

std::atomic<double> masterClock;
std::atomic<double> masterClockTime;

SDL_Window * window;
SDL_Surface * screen;

double getTime() {
	return av_gettime() * 1e-6;
}

#ifndef constexpr
#define constexpr const
#endif

template <typename T>
struct isChronoDuration {
	static constexpr bool value = false;
};

template <typename Rep, typename Period>
struct isChronoDuration<std::chrono::duration<Rep, Period>> {
	static constexpr bool value = true;
};

template <typename T>
class TQueue {
private:
	std::queue<T> _queue;
	std::mutex _mutex;
	std::condition_variable _cond;
	size_t _size;
	
public:
	TQueue(size_t size = 4) : _size(size) {}
	
	void lock(std::unique_lock<decltype(_mutex)> & lock) {
		lock = std::unique_lock<decltype(_mutex)>(_mutex);
	}
	
	bool tryLock(std::unique_lock<decltype(_mutex)> & lock) {
		lock = std::unique_lock<decltype(_mutex)>(_mutex, std::defer_lock);
		return lock.try_lock();
	}
	
	void unlock(std::unique_lock<decltype(_mutex)> & lock) {
		lock.unlock();
		_cond.notify_one();
	}
	
	// Returns True if t is an item from the queue, False otherwise.
	template <typename Duration = std::chrono::milliseconds, int duration_value = 250>
	bool pop(T & t) {
		static_assert(isChronoDuration<Duration>::value, "duration must be a std::chrono::duration");
		
		std::unique_lock<decltype(_mutex)> lock(_mutex);
		
		while (_queue.empty()) {
			if (!_running) {return false;}
			if (_cond.wait_for(lock, Duration(duration_value)) == std::cv_status::timeout) {return false;}
		}
		
		bool p = pop(lock, t);
		
		lock.unlock();
		_cond.notify_one();
		
		return p;
	}
	
	template <typename Duration = std::chrono::milliseconds, int duration_value = 250>
	bool pop(std::unique_lock<decltype(_mutex)> & lock, T & t) {
		static_assert(isChronoDuration<Duration>::value, "duration must be a std::chrono::duration");
		
		if (!lock.owns_lock()) {
			return false;
		}
		
		if (_queue.empty()) {
			return false;
		}
		
		t = _queue.front();
		_queue.pop();
		
		return true;
	}
	
	// Returns True is f was added to the queue, False otherwise
	template <typename Duration = std::chrono::milliseconds, int duration_value = 250>
	bool push(const T & t) {
		static_assert(isChronoDuration<Duration>::value, "duration must be a std::chrono::duration");
		
		std::unique_lock<decltype(_mutex)> lock(_mutex);
		
		while (_queue.size() >= _size) {
			if (!_running) {return false;}
			_cond.wait_for(lock, Duration(duration_value));
		}
		
		_queue.push(t);
		
		lock.unlock();
		_cond.notify_one();
		
		return true;
	}
};

struct StreamContext {
	AVStream * _stream = nullptr;
	AVCodecContext * _codecContext = nullptr;
	int(* _decodeFunc)(AVCodecContext *, AVFrame *, int *, const AVPacket *) = nullptr;
	
	TQueue<AVPacket *> * _packetQueue = nullptr;
	TQueue<AVFrame *> * _frameQueue = nullptr;
	
	bool _enabled = false;
	
	void * _opaque;
	
	std::thread _decodeThread;
};

struct RescaleContext {
	struct SwrContext * _swrContext;
	
	enum AVSampleFormat _sampleFormat;
	int _frequency;
	int _channels;
};

struct ScaleContext {
	struct SwsContext * _swsContext;
	
	int _width;
	int _height;
	
	int _stride[4];
};

std::map<int, struct StreamContext *> _contextMap;

AVPacket * allocPacket() {
	AVPacket * p = (AVPacket *)av_malloc(sizeof(AVPacket));
	if (p == nullptr) {throw std::runtime_error("Error while calling av_malloc");}
	av_init_packet(p);
	return p;
}

AVFrame * allocFrame() {
	AVFrame * f = av_frame_alloc();
	if (f == nullptr) {throw std::runtime_error("Error while calling av_frame_alloc");}
	return f;
}

void decodeThread(struct StreamContext * sc) {
	AVFrame * f = allocFrame();
	
	while (_running) {
		AVPacket * p;
		
		if (!sc->_packetQueue->pop(p)) {continue;}
		
		int frameAvailable = 1;
		while (frameAvailable) { // Keep reading from the same packet so long as new frames are coming, keep going even when AVPacket::size == 0 but with AVPacket::data = nullptr to flush any buffers built by avcodec_decode_video2.
			int read = sc->_decodeFunc(sc->_codecContext, f, &frameAvailable, p);
			if (read < 0) {
				break; // Move on gracefully to next packet in the case of a decode error.
			}
			
			if (frameAvailable) {
				if (!sc->_frameQueue->push(f)) {frameAvailable = 0; continue;}
				f = allocFrame();
			}
			
			p->data += read; // Keep updating the packet based on the last number of read bytes by the decoder.
			p->size -= read;
			if (p->size < 1) {p->data = nullptr;} // Set the data to nullptr when the packet is exausted to flush any buffers created.
		}
		
		av_free_packet(p);
		av_free(p);
	}
	
	av_frame_free(&f);
}

void displayThread(struct StreamContext * sc) {
	auto scale = (struct ScaleContext *)sc->_opaque;
	
	size_t sizeOfFrame = scale->_stride[0] * scale->_height;
	
	auto pic = (uint8_t *)malloc(sizeOfFrame);
	if (pic == nullptr) {
		throw std::runtime_error("malloc failed");
	}
	
	while (_running) {
		AVFrame * f;
		if (!sc->_frameQueue->pop(f)) {continue;}
		
		double base = av_q2d(sc->_stream->time_base);
		double pts = (av_frame_get_best_effort_timestamp(f) * base) + (f->repeat_pict * base * 0.5);
		double delta = pts - masterClock + (getTime() - masterClockTime);
		double deltaClamp = std::min(std::max(delta, 0.0), 1.0);
		
		std::cout << deltaClamp << "\n";
		
		std::this_thread::sleep_for(std::chrono::duration<float>(deltaClamp));
		
		sws_scale(scale->_swsContext, f->data, f->linesize, 0, f->height, &pic, &(scale->_stride[0]));
		
		auto surf = SDL_CreateRGBSurfaceFrom(pic, scale->_width, scale->_height, 24, scale->_stride[0], 0, 0, 0, 0);
		
		SDL_BlitSurface(surf, NULL, screen, NULL);
		
		SDL_FreeSurface(surf);
		
		av_frame_free(&f);
	}
	
	free(pic);
}

// SDL guarantees that this function will not be reentered.
void audioCallback(void * userdata, Uint8 * const stream, int len) {
	StreamContext * sc = (StreamContext *)userdata;
	RescaleContext * rc = (RescaleContext *)sc->_opaque;
	
	memset(stream, 0, len);
	
	uint8_t ** in = nullptr, * out = nullptr;
	int inSamples = 0;
	int outSamples = 0;
	static int outSampleFactor = av_samples_get_buffer_size(NULL, rc->_channels, 1, rc->_sampleFormat, 1); // There may be a better way of doing this.
	if (outSampleFactor > 0) {
		outSamples = len / outSampleFactor;
	} else {
		std::cerr << "av_samples_get_buffer_size error\n";
		return;
	}
	
	std::unique_lock<std::mutex> lock;
	if (!sc->_frameQueue->tryLock(lock)) {return;}
	
	if (av_samples_alloc(&out, nullptr, rc->_channels, outSamples, rc->_sampleFormat, 1) < 0) {
		std::cerr << "av_samples_alloc error\n";
		return;
	}
	
	uint8_t * index = out;
	
	double pts = -1.0;
	static double lastPts = 0.0;
	
	while (outSamples > 0) {
		AVFrame * f = nullptr;
		
		int64_t delay = swr_get_delay(rc->_swrContext, rc->_frequency); // Return the number of samples (in output frequency) buffered in the convert context from last time.
		if (delay < 1) { // No delay, get new frame
			if (!sc->_frameQueue->pop(lock, f)) {break;} // No more frames in the queue, give up rather than waiting for more data.
			in = f->data;
			inSamples = f->nb_samples;
			
			double p = av_frame_get_best_effort_timestamp(f) * av_q2d(sc->_codecContext->time_base);
			
			if (pts < 0.0) {
				pts = p;
				
				masterClock = pts;
				masterClockTime = getTime();
			}
			
			lastPts = p;
		} else {
			in = nullptr;
			inSamples = 0;
			
			/*if (pts < 0.0) {
				double d = ((double)delay / (double)rc->_frequency);
				pts = lastPts + d;
				
				masterClock = pts;
				masterClockTime = av_gettime() * 1e-6;
			}*/
		}
		
		int samples = swr_convert(rc->_swrContext, &index, outSamples, (const uint8_t **)in, inSamples);
		if (samples < 0) { // Convert error
			av_frame_free(&f);
			break;
		}
		outSamples -= samples;
		index += samples * outSampleFactor;
		
		av_frame_free(&f); // Null checked internally.
	}
	
	sc->_frameQueue->unlock(lock);
	
	SDL_MixAudio(stream, out, (Uint32)(index - out), SDL_MIX_MAXVOLUME * 1.0f);
	//memcpy(stream, out, index - out);
	
	av_freep(&out);
}

SDL_AudioFormat AVSampleFormatToSDLAudioFormat(AVSampleFormat format) {
	switch (format) {
		case AV_SAMPLE_FMT_U8:
		case AV_SAMPLE_FMT_U8P:
			return AUDIO_U8; break;
		case AV_SAMPLE_FMT_S16:
		case AV_SAMPLE_FMT_S16P:
			return AUDIO_S16SYS; break;
		case AV_SAMPLE_FMT_S32:
		case AV_SAMPLE_FMT_S32P:
			return AUDIO_S32SYS; break;
		case AV_SAMPLE_FMT_FLT:
		case AV_SAMPLE_FMT_FLTP:
		case AV_SAMPLE_FMT_DBL:
		case AV_SAMPLE_FMT_DBLP:
			return AUDIO_F32SYS; break;
		case AV_SAMPLE_FMT_NONE:
		default:
			throw std::runtime_error("Unknown sample format or no conversion available");
	}
}

AVSampleFormat SDLAudioFormatToAVSampleFormat(SDL_AudioFormat format) {
	switch (format) {
		case AUDIO_U8: return AV_SAMPLE_FMT_U8; break;
		case AUDIO_S16SYS: return AV_SAMPLE_FMT_S16; break;
		case AUDIO_S32SYS: return AV_SAMPLE_FMT_S32; break;
		case AUDIO_F32SYS: return AV_SAMPLE_FMT_FLT; break;
		default: throw std::runtime_error("Unknown sample format or no conversion available"); break;
	}
}

void audioSetup(struct StreamContext * sc) {
	SDL_Init(SDL_INIT_AUDIO);
	
	SDL_AudioSpec wanted_spec, spec;
	wanted_spec.freq =  sc->_stream->codec->sample_rate;
	wanted_spec.format = AVSampleFormatToSDLAudioFormat(sc->_stream->codec->sample_fmt);
	wanted_spec.channels = sc->_stream->codec->channels;
	wanted_spec.silence = 0;
	wanted_spec.samples = SDL_AUDIO_BUFFER_SIZE;
	wanted_spec.callback = audioCallback;
	wanted_spec.userdata = sc;
	
	if (SDL_OpenAudio(&wanted_spec, &spec) < 0) {
		std::cerr << "SDL_OpenAudio: " << SDL_GetError() << "\n";
		return;
	}
	
	auto rc = new RescaleContext();
	rc->_frequency = spec.freq;
	rc->_sampleFormat = SDLAudioFormatToAVSampleFormat(spec.format);
	rc->_channels = spec.channels;
	
	if (!sc->_stream->codec->channel_layout) {
		if (sc->_stream->codec->channels == 2) {
			sc->_stream->codec->channel_layout = AV_CH_LAYOUT_STEREO; // If not channel layout set then make a reasonable assumption as to what it might be.
		} else {
			throw std::runtime_error("Cannot assume reasonable input channel layout.");
		}
	}
	
	int64_t outChannelLayout = 0;
	switch (rc->_channels) {
		case 2: outChannelLayout = AV_CH_LAYOUT_STEREO; break;
		case 4: outChannelLayout = AV_CH_LAYOUT_QUAD; break;
		case 6: outChannelLayout = AV_CH_LAYOUT_5POINT1; break;
		case 8: outChannelLayout = AV_CH_LAYOUT_7POINT1; break;
		default: throw std::runtime_error("Cannot assume reasonable output channel layout."); break;
	}
	
	auto rescale = swr_alloc_set_opts(nullptr, sc->_stream->codec->channel_layout, rc->_sampleFormat, sc->_stream->codec->sample_rate, sc->_stream->codec->channel_layout, sc->_stream->codec->sample_fmt, sc->_stream->codec->sample_rate, 0, nullptr);
	int err = swr_init(rescale);
	if (err) {
		throw std::runtime_error("swr_init error");
	}
	
	rc->_swrContext = rescale;
	sc->_opaque = rc;
}

void videoSetup(struct StreamContext * sc) {
	auto outputPixFmt = av_get_pix_fmt("bgr24");
	auto outputPixFmtDesc = av_pix_fmt_desc_get(outputPixFmt);
	
	auto cc = sc->_codecContext;
	
	auto convert = sws_getContext(cc->width, cc->height, cc->pix_fmt, 1280, 720, outputPixFmt, SWS_BILINEAR, NULL, NULL, NULL);
	if (convert == nullptr) {
		throw std::runtime_error("sws_getContext error");
	}
	
	struct ScaleContext * scale = new ScaleContext();
	scale->_swsContext = convert;
	scale->_width = cc->width;
	scale->_height = cc->height;
	scale->_stride[0] = cc->width * (outputPixFmtDesc->comp[0].step_minus1 + 1);
	
	sc->_opaque = scale;
}

std::string ffmpegError(int errnum) {
	std::string str;
	str.reserve(1024);
	av_strerror(errnum, &str[0], str.capacity());
	return str;
}

int main(int argc, char * argv[]) {
	std::thread videoThread;
	
	SDL_Init(SDL_INIT_VIDEO);
	
	window = SDL_CreateWindow("VideoWindow", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 1280, 720, 0);
	if (window == nullptr) {
		std::cout << "Could not create window: " << SDL_GetError() << "\n";
		return 1;
	}
	screen = SDL_GetWindowSurface(window);
	
	int err = 0;
	
	av_register_all();
	
	av_dict_set(&_options, "timeout", "100", 0);
	
	if ((err = avformat_open_input(&_format, argv[1], nullptr, &_options)) < 0) {
		std::cerr << "Error while calling avformat_open_input (probably invalid file format)" << "\n";
		return 1;
	}
	
	if (avformat_find_stream_info(_format, nullptr) < 0) {
		std::cerr << "Error while calling avformat_find_stream_info\n";
		return 1;
	}
	
	for (unsigned int i = 0; i < _format->nb_streams; ++i) {
		auto istream = _format->streams[i]; // pointer to a structure describing the stream
		
		const auto codec = avcodec_find_decoder(istream->codec->codec_id);
		if (codec == nullptr) {
			std::cerr << "Codec required by video stream not available\n";
			continue;
		}
		
		const auto codecContext = avcodec_alloc_context3(codec);
		if (codecContext == nullptr) {
			throw std::runtime_error("avcodec_alloc_context3 failed\n"); // This error is too fatal to recover from, out of memory assumed and will only cause further failures down the line.
		}
		
		decltype(StreamContext::_decodeFunc) decodeFunc = nullptr;
		
		switch (istream->codec->codec_type) { // the type of data in this stream, notable values are AVMEDIA_TYPE_VIDEO and AVMEDIA_TYPE_AUDIO
			case AVMEDIA_TYPE_VIDEO: {
				codecContext->pix_fmt = istream->codec->pix_fmt; // Required for raw video.
				codecContext->width = istream->codec->width;
				codecContext->height = istream->codec->height;
				codecContext->extradata = istream->codec->extradata;
				codecContext->extradata_size = istream->codec->extradata_size;
				
				decodeFunc = avcodec_decode_video2;
			} break;
			case AVMEDIA_TYPE_AUDIO: {
				codecContext->channels = 2; // Set some parameters.
				codecContext->sample_rate = 44100;
				codecContext->channel_layout = AV_CH_LAYOUT_STEREO;

				decodeFunc = avcodec_decode_audio4;
			} break;
		}
		
		if (avcodec_open2(codecContext, codec, nullptr) < 0) {
			std::cerr << "Could not open video codec\n";
			avcodec_close(codecContext);
			continue;
		}
		
		// Leak boundary.
		
		auto streamContext = new StreamContext();
		streamContext->_stream = istream;
		streamContext->_codecContext = codecContext;
		streamContext->_decodeFunc = decodeFunc;
		
		_contextMap[istream->index] = streamContext;
	}
	
	// After getting the information for all the streams make a choice as to which to decode, frequently the first video and audio stream.
	// Also, create the audio device.
	for (decltype(_contextMap)::iterator i = _contextMap.begin(); i != _contextMap.end(); ++i) {
		auto sc = i->second;
		auto t = sc->_stream->codec->codec_type;
		if (t == AVMEDIA_TYPE_VIDEO) {
			videoSetup(sc);
			
			sc->_enabled = true; // Only enable decoding of audio and video streams for now.
			
			sc->_packetQueue = new TQueue<AVPacket *>(32);
			sc->_frameQueue = new TQueue<AVFrame *>(32);
			sc->_decodeThread = std::thread(decodeThread, sc);
			
			videoThread = std::thread(displayThread, sc);
		} else if (t == AVMEDIA_TYPE_AUDIO) {
			audioSetup(sc);
			
			sc->_enabled = true; // Only enable decoding of audio and video streams for now.
			
			sc->_packetQueue = new TQueue<AVPacket *>(32);
			sc->_frameQueue = new TQueue<AVFrame *>(32);
			sc->_decodeThread = std::thread(decodeThread, sc);
		}
	}
	
	AVPacket * packet = allocPacket();
	
	SDL_PauseAudio(0);
	
	masterClock = 0.0;
	masterClockTime = getTime();
	
	while (_running && av_read_frame(_format, packet) == 0) {
		decltype(_contextMap)::const_iterator i;
		if ((i = _contextMap.find(packet->stream_index)) != _contextMap.end()) {
			if (i->second->_enabled) {
				i->second->_packetQueue->push(packet);
				packet = allocPacket();
			} else {
				av_free_packet(packet);
			}
		} else {
			av_free_packet(packet);
		}
		
		// Seeking
		
		SDL_Event event;
		while (SDL_PollEvent(&event)) {
			switch (event.type) {
				case SDL_QUIT:
					_running = false;
					break;
			}
		}
		
		SDL_UpdateWindowSurface(window);
	}
	
	SDL_PauseAudio(1);
	
	av_free(packet);
	
	_running = false;
	
	for (decltype(_contextMap)::iterator i = _contextMap.begin(); i != _contextMap.end(); ++i) {
		auto sc = i->second;
		
		if (sc->_stream->codec->codec_type == AVMEDIA_TYPE_AUDIO) {
			auto scale = (struct ScaleContext *)sc->_opaque;
			sws_freeContext(scale->_swsContext);
			delete scale;
		} else if (sc->_stream->codec->codec_type == AVMEDIA_TYPE_AUDIO) {
			auto rc = (struct RescaleContext *)sc->_opaque;
			swr_close(rc->_swrContext);
			delete rc;
		}
		
		if (sc->_decodeThread.joinable()) {sc->_decodeThread.join();}
		
		avcodec_close(sc->_codecContext);
		avcodec_free_context(&(sc->_codecContext));
		
		delete sc->_frameQueue; // Clear up queue, may be null, C++ says that should be ok.
		delete sc->_packetQueue;
		delete sc;
	}
	
	if (videoThread.joinable()) {
		videoThread.join();
	}
	
	avformat_close_input(&_format);
	avformat_free_context(_format);
	
	SDL_DestroyWindow(window);
	SDL_Quit();
	
    return 0;
}
