"""Bounded product execution. Raw GraphModel and exports retain PyTorch semantics."""
import torch

if __package__:
    from .bag_source import execute_bag_source, normalize_bag_source, source_layouts
    from .embedding_bag_packet import EmbeddingBagPacket
else:
    from bag_source import execute_bag_source, normalize_bag_source, source_layouts
    from embedding_bag_packet import EmbeddingBagPacket

class ProductExecutionAdapter:
    def __init__(self,model):
        self.model=model
        sources=[i for i in model.order if model.nodes[i]['type'] in ('input','index_input','bag_input')]
        self.named_inputs=getattr(model,'named_inputs',{})
        if not self.named_inputs and len(sources)!=1:raise ValueError('Product execution requires exactly one supported model source.')
        self.input_id=sources[0]
        self._bag_configuration=(normalize_bag_source(model.nodes[self.input_id]['config'])
                                 if model.nodes[self.input_id]['type']=='bag_input' else None)
        if self._bag_configuration is None:
            self.input_shape=model.layouts[self.input_id].sample
            self.input_dtype=getattr(torch,model.dtypes[self.input_id])
            self.features=model.widths.get(self.input_id)
        else:
            # Packed item coordinates have no observation-first sample shape.
            self.input_shape=None;self.input_dtype=None;self.features=None
            self._validate_bag_source_contract()
        self.objective_context=getattr(model,'objective_context',None)
        self.device=next(model.parameters()).device
        if self.device.type not in ('cpu','mps'):raise ValueError('Product execution requires CPU or MPS; select a supported model device.')
        self.validate_parameters()

    @property
    def input_contract(self):
        if self.named_inputs:return {'kind':'namedInputs','schemaVersion':1,'inputs':{name:{'nodeID':node,'dtype':self.model.dtypes[node],'sampleShape':list(self.model.layouts[node].sample)} for name,node in self.named_inputs.items()}}
        if self._bag_configuration is None:
            return {'dtype':str(self.input_dtype).removeprefix('torch.'),'sampleShape':list(self.input_shape)}
        self._validate_bag_source_contract()
        return {'kind':'bagPacket','schemaVersion':1,'sourceID':self.input_id,
                'configuration':dict(self._bag_configuration),
                'ports':{role:{'role':role,'dtype':'float32' if role=='per_sample_weights' else 'int64'}
                         for role in source_layouts(self.input_id,self._bag_configuration)}}

    def _validate_bag_source_contract(self):
        node=self.model.nodes.get(self.input_id)
        if (type(node) is not dict or node.get('type')!='bag_input'
                or normalize_bag_source(node.get('config'))!=self._bag_configuration):
            raise ValueError('Bag input source controls differ from the captured product contract.')
        expected=source_layouts(self.input_id,self._bag_configuration)
        expected_types={role:'float32' if role=='per_sample_weights' else 'int64' for role in expected}
        if (type(self.model.layouts.get(self.input_id)) is not dict
                or self.model.layouts[self.input_id]!=expected
                or type(self.model.dtypes.get(self.input_id)) is not dict
                or self.model.dtypes[self.input_id]!=expected_types):
            raise ValueError('Bag input captured source roles, geometry or dtypes differ from their coherent contract.')

    @staticmethod
    def _bag_rows(packet):
        return packet.indices.shape[0] if packet.layout=='fixed' else packet.offsets.numel()-int(packet.include_last_offset)

    def observation_count(self,x):
        self.validate_input(x)
        return self._bag_rows(x) if self._bag_configuration is not None else len(x)

    def _validate_bag_input(self,x,*,device,context,rows=None):
        self._validate_bag_source_contract()
        execute_bag_source(x,self._bag_configuration)
        target_device=torch.device(device)
        if target_device.type not in ('cpu','mps') or x.indices.device!=target_device:
            raise ValueError(f'{context}: expected device {target_device}; move the packet to the selected device.')
        if rows is not None and self._bag_rows(x)!=rows:
            raise ValueError(f'{context}: expected {rows} bag observations.')
        return x

    @staticmethod
    def validate_tensor(value,*,features=None,sample_shape=None,device,context,rows=None,check_finite=True,dtype=torch.float32):
        def reject(reason):raise ValueError(f'{context}: {reason}')
        if not isinstance(value,torch.Tensor):reject('provide a tensor.')
        if value.is_nested or value.layout!=torch.strided:reject('provide a dense strided tensor; sparse/nested layouts are unsupported.')
        expected=tuple(sample_shape) if sample_shape is not None else (features,)
        if value.ndim!=len(expected)+1 or value.shape[0]==0 or tuple(value.shape[1:])!=expected:reject(f'expected a nonempty batch with sample shape {expected}.')
        if rows is not None and value.shape[0]!=rows:reject(f'expected {rows} observations.')
        if value.dtype!=dtype:reject(f'expected {str(dtype).removeprefix("torch.")}; convert the source explicitly before execution.')
        if value.device!=torch.device(device):reject(f'expected device {device}; move the input to the selected model device.')
        if check_finite and not torch.isfinite(value).all():reject('values must be finite; inspect feature scaling or the producing component.')
        return value

    def validate_parameters(self):
        for name,value in self.model.named_parameters():
            if value.is_nested or value.layout!=torch.strided or value.dtype!=torch.float32 or value.device!=self.device or not torch.isfinite(value).all():raise ValueError(f'Model parameter {name}: expected finite dense float32 values on {self.device}; reload compatible weights.')

    def validate_input(self,x,*,check_finite=True,device=None,context='Model input'):
        target_device=self.device if device is None else device
        if self.named_inputs:
            if type(x) is not dict or set(x)!=set(self.named_inputs):raise ValueError('Provide exactly the declared named inputs.')
            from_model_input=__import__((__package__+'.model_input') if __package__ else 'model_input',fromlist=['observation_count'])
            rows=from_model_input.observation_count(x)
            for name,node in self.named_inputs.items():self.validate_tensor(x[name],sample_shape=self.model.layouts[node].sample,dtype=getattr(torch,self.model.dtypes[node]),device=target_device,context=context+' / '+name,rows=rows,check_finite=check_finite)
            return x
        if self._bag_configuration is not None:
            # Packet geometry/finite item weights are never a trusted-path exemption.
            return self._validate_bag_input(x,device=target_device,context=context)
        return self.validate_tensor(x,sample_shape=self.input_shape,dtype=self.input_dtype,device=target_device,context=context,check_finite=check_finite)

    def validate_output(self,value,*,context='Model output',rows=None,features=None):
        if self.objective_context is not None:
            return self.validate_tensor(value,sample_shape=self.model.layouts[self.model.output_id].sample,dtype=getattr(torch,self.model.dtypes[self.model.output_id]),device=self.device,context=context,rows=rows)
        return self.validate_tensor(value,features=features if features is not None else self.model.widths[self.model.output_id],device=self.device,context=context,rows=rows)

    def validate_stage(self,value,ident,*,rows=None,device=None):
        layouts = self.model.layouts[ident]
        dtypes = self.model.dtypes[ident]
        context = f'Captured component {ident}'
        target_device = self.device if device is None else device
        if ident==self.input_id and self._bag_configuration is not None:
            self._validate_bag_source_contract()
            if type(value) is not dict or set(value)!=set(layouts):
                raise ValueError(f'{context}: provide exactly the declared bag coordinate ports {sorted(layouts)}.')
            packet=EmbeddingBagPacket(self._bag_configuration['layout'],value['indices'],value.get('offsets'),
                                      value.get('per_sample_weights'),self._bag_configuration['include_last_offset'])
            self._validate_bag_input(packet,device=target_device,context=context,rows=rows)
            return value
        if isinstance(layouts, dict) or isinstance(dtypes, dict):
            if (type(layouts) is not dict or type(dtypes) is not dict or not layouts
                    or set(layouts) != set(dtypes) or any(type(port) is not str or not port for port in layouts)):
                raise ValueError(f'{context}: output contract port sets must agree.')
            if type(value) is not dict or set(value) != set(layouts):
                raise ValueError(f'{context}: provide exactly the declared output ports {sorted(layouts)}.')
            for port, layout in layouts.items():
                self.validate_tensor(value[port], sample_shape=layout.sample,
                    dtype=getattr(torch,dtypes[port]), device=target_device,
                    context=f'{context} output {port}', rows=rows)
            return value
        return self.validate_tensor(value,sample_shape=layouts.sample,dtype=getattr(torch,dtypes),
            device=target_device,context=context,rows=rows)

    def forward(self,x,inspect=False,operation_observer=None):
        self.validate_input(x)
        rows=self._bag_rows(x) if self._bag_configuration is not None else next(iter(x.values())).shape[0] if self.named_inputs else len(x)
        if operation_observer is None:
            result=self.model(x,inspect=inspect)
        else:
            # Preserve the original forward calling convention for wrappers/hooks.
            # The worker owns this model exclusively; restore even on refusal/cancel.
            had_observer=hasattr(self.model,'_capture_operation_observer')
            previous=getattr(self.model,'_capture_operation_observer',None)
            self.model._capture_operation_observer=operation_observer
            try:result=self.model(x,inspect=inspect)
            finally:
                if had_observer:self.model._capture_operation_observer=previous
                else:delattr(self.model,'_capture_operation_observer')
        if inspect:
            output,values=result
            for ident,value in values.items():self.validate_stage(value,ident,rows=rows)
        else:output=result
        self.validate_output(output,rows=rows)
        return result

    def _forward_trusted(self,x,*,inspect=False):
        """Worker-only hot path: source was fully checked; preserve cheap metadata guards."""
        self.validate_input(x,check_finite=False)
        return self.model(x,inspect=True) if inspect else self.model(x)


def execution_contract(runtime_version,available_devices):
    """Fresh JSON descriptor of implemented product policies, never catalog coverage."""
    return {
        'schemaVersion':1,'adapterID':'kwip.runner.product-execution.v1',
        'runtimeVersion':runtime_version,'graphSchemaVersions':[1,2],
        'tensor':{'rank':2,'dtype':'float32','layout':'denseStrided','noncontiguous':True,'minimumBatch':1,'preservesRows':True},
        'supportedDevices':['cpu','mps'],'availableDevices':list(available_devices),'biasModes':[False,True],
        'task':{'kind':'binaryClassification','inputFeatures':2,'outputFeatures':2},
        'limits':{'nodes':128,'edges':256,'parameters':2000000,'configuredWidth':512,'intermediateWidth':4096},
        'operations':{'inference':'boundedGraph','training':'experimentOwnedAdam','exactResume':'cpuEpochBoundary','fineTune':'weightsOnlyFreshOptimizer','export':'pythonSourceRawTorch'},
        'finiteChecks':['sourcePartitions','initialParameters','publicForwardInputsOutputs','inspectedStages','validationLogitsAndLoss','trainingLoss'],
        'rawModelSemantics':'pytorch',
    }
