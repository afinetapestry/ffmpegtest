Threaded FFmpeg Example
=======================

This repository contained a minimal working example of how to decode media files using FFmpeg and present the video and audio using SDL.

This example is different because each of the decoding functions are neatly seperated into their own threads with the packets and frames being passed around via threadsafe queues.
