

#### install 

```shell
conda create -p /media/gpt4-pdf-chatbot-langchain/pyenv-loom-zhipuai  python=3.10  
conda activate /media/gpt4-pdf-chatbot-langchain/pyenv-loom-zhipuai

```

#### core
```shell

pip install loom_core-0.1.0-py3-none-any.whl  --force-reinstall         
```

#### run
```shell

python -m loom_core.openai_plugins.deploy.subscribe  -f /media/gpt4-pdf-chatbot-langchain/loom-openai-plugins-zhipuai/loom.yaml
```
