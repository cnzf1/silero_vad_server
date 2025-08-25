.PHONY: run

run:
	VAD_LISTEN=0.0.0.0:9094 nohup target/release/silero_vad_server >logs/debug.log 2>&1 &
