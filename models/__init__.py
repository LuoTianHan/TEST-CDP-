

# 模型注册表
_MODEL_REGISTRY = {
    'TEST_CDP_Llama': 'models.TEST_CDP_Llama.Model',
}


def get_model(model_name, configs):
    """
    获取模型实例（延迟加载）
    :param model_name: 模型名称
    :param configs: 模型配置
    :return: 模型实例
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"未知模型: {model_name}, 可用模型: {list(_MODEL_REGISTRY.keys())}")
    
    module_path, class_name = _MODEL_REGISTRY[model_name].rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)
    return model_class(configs)



__all__ = ['get_model']
