# An API server for AI VAD

This is the VAD server that is designed to wok with the [EchoKit server](https://github.com/second-state/echokit_server).

## Install libtorch dependencies

Regular Linux CPU

```
# download libtorch
curl -LO https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu124.zip

unzip libtorch-cxx11-abi-shared-with-deps-2.4.0+cu124.zip

# Add to ~/.zprofile or ~/.bash_profile
export LD_LIBRARY_PATH=$(pwd)/libtorch/lib:$LD_LIBRARY_PATH
export LIBTORCH=$(pwd)/libtorch 
```

## Build the API server

```
git clone https://github.com/second-state/silero_vad_server

cd silero_vad_server
cargo build --release
```

## Run the API server

```
VAD_LISTEN=0.0.0.0:9094 nohup target/release/silero_vad_server &
```

In the [EchoKit server](https://github.com/second-state/echokit_server) configuration, you can now use the VAD server in the `[asr]` section to use it together with any ASR API.

```
[asr]
url = "http://localhost:9092/v1/audio/transcriptions"
lang = "auto"
prompt = "Hello\n你好\n(noise)\n(bgm)\n(silence)\n"
vad_url = "http://localhost:9094/v1/audio/vad"
```
