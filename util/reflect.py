from torchvision.transforms import Compose
from torch.utils.data import DataLoader


def fetch_class(query):
    *module, _class = query.split(".")
    mod = __import__(".".join(module), fromlist=[_class])
    return getattr(mod, _class)


def init_class(config, *args, **kwargs):
    """Given a class definition dictionary, instantiate a class"""
    _class = fetch_class(config["name"])
    return _class(*config.get("args", []), *args, **config.get("kwargs", {}), **kwargs)


def init_data(config):
    """Given a PyTorch dataset configuration dictionary, instantiate a dataset and dataloader"""
    ds_class = fetch_class(config["name"])
    d_transform = list(map(init_class, config.get("transform", [])))
    d_ttransform = list(map(init_class, config.get("target_transform", [])))
    ds = ds_class(
        *config.get("args", []),
        **config.get("kwargs", {}),
        transform=Compose(d_transform) if d_transform else None,
        target_transform=Compose(d_ttransform) if d_ttransform else None
    )
    dl = DataLoader(ds, **config.get("dataloader_kwargs", {}))
    return ds, dl
