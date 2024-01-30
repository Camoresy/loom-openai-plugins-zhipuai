

#### install 

```shell
conda create -p /media/gpt4-pdf-chatbot-langchain/pyenv-loom-zhipuai  python=3.10  
conda activate /media/gpt4-pdf-chatbot-langchain/pyenv-loom-zhipuai

```

#### core
```shell

pip install loom_core-0.1.0-py3-none-any.whl  --force-reinstall         
```

#### openai_plugins[loom.yaml](loom.yaml)
配置文件，详细说明见[Loom](https://github.com/LMMVA/LooM/blob/master/src%2Fcore%2FREADME.md)

#### 远程注册
```shell

python -m loom_core.openai_plugins.deploy.subscribe  -f /media/gpt4-pdf-chatbot-langchain/loom-openai-plugins-zhipuai/loom.yaml
```

#### 本地运行
```shell
python -m loom_core.openai_plugins.deploy.local  -f /media/gpt4-pdf-chatbot-langchain/loom-openai-plugins-zhipuai/loom.yaml
```
