from pyro.infer import TraceGraph_ELBO


class CustomELBO(TraceGraph_ELBO):
    """
    Taken from DSCM repo
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.trace_storage = {'model': None, 'guide': None}

    def _get_trace(self, model, guide, args, kwargs):
        model_trace, guide_trace = super()._get_trace(model, guide, args, kwargs)

        self.trace_storage['model'] = model_trace
        self.trace_storage['guide'] = guide_trace

        return model_trace, guide_trace
