import Foundation
import KwipMLCore

/// Worker identity differs from the native workflow run identity. Values must
/// come from the archived, hash-verified parent lifecycle files.
struct WorkflowLifecycleParentEvidence:Equatable {
    let workerRunID:String
    let checkpointSHA256:String
    let completedEpoch:Int
    let hasScheduler:Bool
    let hasObjective:Bool
    let gradientPolicy:GradientPolicyConfiguration?
    let trainability:TrainabilityConfiguration?
    let gradientIdentity:WorkflowGradientPolicyIdentity?
    let dataLoadingIdentity:WorkflowDataLoadingIdentity?
    var hasDataLoading:Bool {dataLoadingIdentity != nil}
    var hasGradientPolicy:Bool {gradientPolicy != nil}
    var hasTrainability:Bool {trainability != nil}
    init(workerRunID:String,checkpointSHA256:String,completedEpoch:Int,hasScheduler:Bool=false,hasObjective:Bool=false,gradientPolicy:GradientPolicyConfiguration?=nil,trainability:TrainabilityConfiguration?=nil,gradientIdentity:WorkflowGradientPolicyIdentity?=nil,dataLoadingIdentity:WorkflowDataLoadingIdentity?=nil) {
        self.workerRunID=workerRunID;self.checkpointSHA256=checkpointSHA256;self.completedEpoch=completedEpoch
        self.hasScheduler=hasScheduler;self.hasObjective=hasObjective;self.gradientPolicy=gradientPolicy;self.trainability=trainability;self.gradientIdentity=gradientIdentity;self.dataLoadingIdentity=dataLoadingIdentity
    }
}

enum WorkflowLifecycleVerifier {
    /// JSON metadata binds native persistent ownership; worker validates the binary tensors.
    static func validateComposition(_ manifest: [String:JSONValue], graph: GraphDocument, registry: ComponentRegistry, context: GraphValidationContext) throws {
        func fail() -> WorkflowPhaseError { WorkflowPhaseError("Checkpoint composition plan or ordered native state ownership differs from the captured model.") }
        let raw: JSONValue? = {
            guard case .object(let checkpoint) = manifest["checkpoint"], case .object(let contract) = checkpoint["contract"] else { return nil }
            return contract["composition"]
        }()
        guard let plan = graph.compositionPlan else {
            guard raw == nil else { throw fail() }; return
        }
        try plan.validate(graph: graph, registry: registry)
        guard case .object(let composition) = raw, Set(composition.keys) == ["schemaVersion","plan","parameterOwnership","stateOwnership"], composition["schemaVersion"] == .integer(1) else { throw fail() }
        let encodedPlan = try JSONDecoder().decode(JSONValue.self, from: WorkflowArchiveCodec.encode(plan))
        let savedGraph = try manifest["graph"].map { try JSONDecoder().decode(GraphDocument.self, from: WorkflowArchiveCodec.encode($0)) }
        guard savedGraph == graph, composition["plan"] == encodedPlan else { throw fail() }
        let parameters = try optimizerParameterOwnership(graph, context: context, capturedRegistry: registry, initializationOwnership: true)
        let registration = try compositionRegistration(graph, registry: registry)
        let byNode = Dictionary(uniqueKeysWithValues: graph.nodes.map { ($0.id,$0) })
        var state: [JSONValue] = []
        for (nodeID,path) in registration {
            for item in parameters {
                guard case .object(var fields) = item, fields["node"] == .string(nodeID) else { continue }
                fields["kind"] = .string("parameter"); state.append(.object(fields))
            }
            guard let node = byNode[nodeID], StatefulNormalizationConfiguration.componentIDs.contains(node.type) else { continue }
            let values = try GraphMigration.effectiveConfiguration(of: node, schemaVersion: graph.schemaVersion, registry: registry)
            let configuration = try StatefulNormalizationConfiguration(componentID: node.type, values: values)
            guard configuration.tracksRunningStatistics else { continue }
            for (field,shape,dtype) in [("running_mean",[configuration.channelCount],"float32"),("running_var",[configuration.channelCount],"float32"),("num_batches_tracked",[],"int64")] {
                state.append(.object(["node":.string(nodeID),"parameter":.string(field),"name":.string(path+"."+field),"shape":.array(shape.map { .integer(Int64($0)) }),"dtype":.string(dtype),"kind":.string("buffer")]))
            }
        }
        guard composition["parameterOwnership"] == .array(parameters), composition["stateOwnership"] == .array(state) else { throw fail() }
    }
    /// Native module hierarchy is determined by captured registration order, not execution order.
    private static func compositionRegistration(_ graph: GraphDocument, registry: ComponentRegistry) throws -> [(String,String)] {
        guard let plan = graph.compositionPlan else { return [] }
        try plan.validate(graph: graph, registry: registry)
        let rows = Dictionary(uniqueKeysWithValues: plan.containers.map { ($0.id,$0) })
        var result: [(String,String)] = []
        func visit(_ id: String, _ path: String) throws {
            guard let row = rows[id] else { throw WorkflowPhaseError("Composition container is missing.") }
            for child in row.children {
                let native = path + (row.kind == .graph ? ".blocks." : ".") + child.key
                if child.kind == .container { try visit(child.id,native) }
                else { result.append((child.id,native+".operation")) }
            }
        }
        try visit(plan.root,"composition_root")
        return result
    }
    static func validateAttentionExecution(_ runtime:[String:JSONValue],graph:GraphDocument,context:GraphValidationContext = .bundledDataset,capturedRegistry:ComponentRegistry?=nil)throws {
        let nodes=graph.nodes.filter {AttentionConfiguration.componentIDs.contains($0.type)}
        if nodes.isEmpty {guard runtime["attentionExecution"] == nil else {throw WorkflowPhaseError("Saved attention execution policy has no matching model component.")};return}
        guard let device=runtime["device"]?.stringValue,["cpu","mps"].contains(device) else {throw WorkflowPhaseError("Attention execution policy requires the actual supported device.")}
        let registry=try capturedRegistry ?? ComponentRegistry.builtIn(),validation=GraphValidator.validate(graph,registry:registry,context:context)
        let byID=Dictionary(uniqueKeysWithValues:nodes.map {($0.id,$0)})
        var expected:[JSONValue]=[]
        for id in validation.orderedNodeIDs {
            guard let node=byID[id] else {continue}
            let values=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:graph.schemaVersion,registry:registry)
            if device == "mps",node.type == "scaled_dot_product_attention",(values["dropout"]?.doubleValue ?? 0)>0 {
                expected.append(.object(["nodeID":.string(id),"operator":.string(node.type),"computeDevice":.string("mps"),"policy":.string("deviceLocalPublicOps"),"phase":.string("training")]))
            }
        }
        guard Set(nodes.map(\.id)).isSubset(of:Set(validation.orderedNodeIDs)),runtime["attentionExecution"] == (expected.isEmpty ? nil:.array(expected)) else {throw WorkflowPhaseError("Saved attention execution policy differs from the actual component order, device and dropout controls.")}
    }
    /// JSON metadata and immutable checkpoint hashes are native evidence. Torch tensors/update
    /// counters are validated by the production worker; this host never interprets pickle state.
    static func validateOptimizer(_ manifest:[String:JSONValue],expected:OptimizerRecipe?,graph:GraphDocument,expectedConfiguration:OptimizerConfiguration?=nil,context:GraphValidationContext = .preparedWorkflow,capturedRegistry:ComponentRegistry?=nil,gradientIdentity:WorkflowGradientPolicyIdentity?=nil)throws {
        func require(_ valid:Bool)throws {guard valid else {throw WorkflowPhaseError("Optimizer artifacts differ from the captured recipe, effective rate, runtime or ordered learned-parameter ownership.")}}
        guard case .object(let settings)=manifest["settings"],case .object(let provenance)=manifest["provenance"] else {throw WorkflowPhaseError("Optimizer settings or provenance are missing.")}
        let contractOptimizer:JSONValue? = {
            guard case .object(let checkpoint)=manifest["checkpoint"],case .object(let contract)=checkpoint["contract"] else {return nil}
            return contract["optimizer"]
        }()
        if let expected,let expectedConfiguration {try require(expected.configuration == expectedConfiguration)}
        guard let configuration=expected?.configuration ?? expectedConfiguration else {
            try require(settings["optimizerRecipe"] == nil && settings["optimizerRateSource"] == nil && provenance["optimizer"] == nil)
            if let name=settings["optimizer"] {try require(name == .string("Adam"))}
            if let raw=contractOptimizer {
                guard case .object(let legacy)=raw else {throw WorkflowPhaseError("Legacy Adam checkpoint metadata is malformed.")}
                try require(Set(legacy.keys) == ["name","learningRate","betas","eps","weightDecay","amsgrad"] && legacy["name"] == .string("Adam") && legacy["learningRate"]?.doubleValue == settings["learningRate"]?.doubleValue && legacy["eps"]?.doubleValue == 1e-8 && legacy["weightDecay"]?.doubleValue == 0 && legacy["amsgrad"] == .bool(false))
                guard case .array(let betas)=legacy["betas"] else {throw WorkflowPhaseError("Legacy Adam averages are missing.")}
                try require(betas.count == 2 && betas[0].doubleValue == 0.9 && betas[1].doubleValue == 0.999)
            }
            return
        }
        if let expected {try require(expected.source.domain == .control && !expected.source.localID.isEmpty && expected.source.localID.utf8.count <= 128)}
        let names=["optimizer_sgd":"SGD","optimizer_adam":"Adam","optimizer_adamw":"AdamW","optimizer_rmsprop":"RMSprop","optimizer_adagrad":"Adagrad"]
        guard let name=names[configuration.componentID],let rawRecipe=settings["optimizerRecipe"],case .object(let optimizer)=provenance["optimizer"],case .object(let checkpointOptimizer)=contractOptimizer,case .object(let runtime)=manifest["runtime"],let version=runtime["torch"]?.stringValue,!version.isEmpty,version.utf8.count<=128 else {throw WorkflowPhaseError("Explicit optimizer identity or runtime evidence is missing.")}
        func recipeMatches(_ raw:JSONValue?)throws->Bool {
            guard let raw,case .object(let recipe)=raw,case .object(let config)=recipe["config"],Set(config.keys)==Set(configuration.values.keys) else {return false}
            return try OptimizerConfiguration.decodeRecipe(raw) == configuration
        }
        try require(settings["optimizer"] == .string(name) && settings["optimizerRateSource"] == .string("explicitRecipe") && settings["learningRate"]?.doubleValue == configuration.learningRate)
        try require(try recipeMatches(rawRecipe))
        try require(Set(optimizer.keys) == ["recipe","parameterOwnership","implementation","torchVersion"] && optimizer["implementation"] == .string(name) && optimizer["torchVersion"] == .string(version))
        try require(try recipeMatches(optimizer["recipe"]))
        try require(Set(checkpointOptimizer.keys) == ["schemaVersion","recipe","parameterOwnership"] && checkpointOptimizer["schemaVersion"] == .integer(1))
        try require(try recipeMatches(checkpointOptimizer["recipe"]))
        let ownership=try gradientIdentity?.optimizerOwnership ?? optimizerParameterOwnership(graph,context:context,capturedRegistry:capturedRegistry)
        try require(!ownership.isEmpty && optimizer["parameterOwnership"] == .array(ownership) && checkpointOptimizer["parameterOwnership"] == .array(ownership))
    }

    /// Shared bounded shape validation for live ticks and immutable trace records.
    /// Full trace verification additionally proves saved metric and independent curve values.
    static func validateSchedulerRecord(_ record:JSONValue,configuration:SchedulerConfiguration,expectedTick:Int,parameterCount:Int?=nil)throws {
        func fail()->WorkflowPhaseError {WorkflowPhaseError("Scheduler tick contains unsupported fields, timing, rates, momentum or update ownership.")}
        func require(_ valid:Bool)throws {guard valid else {throw fail()}}
        func integer(_ raw:JSONValue?,maximum:Int=SchedulerConfiguration.maximumTicks)throws->Int {guard case .integer(let n)=raw,n>=1,n<=maximum else {throw fail()};return Int(n)}
        func number(_ raw:JSONValue?,negative:Bool=false)throws->Double {guard let n=raw?.doubleValue,n.isFinite,abs(n)<=OptimizerConfiguration.scalarMaximum,(negative || n>=0) else {throw fail()};return n}
        guard case .object(let row)=record else {throw fail()}
        let batch=configuration.componentID == "scheduler_one_cycle_lr",c=configuration.values
        var keys:Set<String>=["tick","epoch","usedRate","nextRate","nextRateUnused"]
        if batch {keys.formUnion(["batch","updatedParameterIndices"]);if c["cycle_momentum"] == .bool(true) {keys.formUnion(["usedMomentum","nextMomentum"])}}
        if configuration.componentID == "scheduler_reduce_on_plateau" {keys.formUnion(["metric","metricValue","changed"])}
        try require(Set(row.keys)==keys && (1...SchedulerConfiguration.maximumTicks).contains(expectedTick))
        try require(integer(row["tick"])==expectedTick)
        let epoch=try integer(row["epoch"])
        let used=try number(row["usedRate"]),next=try number(row["nextRate"],negative:true)
        guard let unused=row["nextRateUnused"]?.boolValue else {throw fail()}
        if batch {
            let horizon=Int(c["planned_total_batches"]!.integerValue!)
            try require(expectedTick<=horizon && unused == (expectedTick==horizon) && (next>=0 || unused))
            _=try integer(row["batch"],maximum:100_000_000)
            guard case .array(let raw)=row["updatedParameterIndices"] else {throw fail()}
            let ceiling=parameterCount ?? 10_000
            try require(ceiling>0 && ceiling<=200_000 && raw.count<=ceiling)
            let indices=try raw.map { item -> Int in guard case .integer(let n)=item,n>=0,n<ceiling else {throw fail()};return Int(n)}
            try require(indices==Array(Set(indices)).sorted())
            if c["cycle_momentum"] == .bool(true) {
                for key in ["usedMomentum","nextMomentum"] {
                    let n=try number(row[key]);try require((0...1).contains(n))
                    if !(unused && key == "nextMomentum") {
                        let low=c["base_momentum"]!.doubleValue!,high=c["max_momentum"]!.doubleValue!
                        let tolerance=max(1e-15,abs(high)*1e-12)
                        try require(n>=low-tolerance && n<=high+tolerance)
                    }
                }
            }
        } else {
            try require(epoch==expectedTick && !unused && next>=0)
            if configuration.componentID == "scheduler_cosine_annealing_lr" {try require(expectedTick<=Int(c["T_max"]!.integerValue!))}
        }
        if configuration.componentID == "scheduler_reduce_on_plateau" {
            try require(row["metric"] == c["metric"])
            let metric=try number(row["metricValue"],negative:true)
            if c["metric"] == .string("validation_accuracy") {try require((0...1).contains(metric))}
            try require(row["changed"] == .bool(used != next))
        }
    }

    /// Verify native JSON identity and independently recomputed rates. The hashed Torch
    /// checkpoint remains opaque here; production worker owns its tensors/native state.
    static func validateScheduler(_ manifest:[String:JSONValue],expected:SchedulerRecipe?,optimizer:OptimizerRecipe?,graph:GraphDocument,trace:JSONValue?,metrics:[[String:JSONValue]],expectedConfiguration:SchedulerConfiguration?=nil,optimizerConfiguration:OptimizerConfiguration?=nil,context:GraphValidationContext = .preparedWorkflow,capturedRegistry:ComponentRegistry?=nil,gradientIdentity:WorkflowGradientPolicyIdentity?=nil)throws {
        func fail()->WorkflowPhaseError {WorkflowPhaseError("Scheduler artifacts differ from the captured recipe, completed validation, timing, rate curve or parameter ownership.")}
        func require(_ value:Bool)throws {guard value else {throw fail()}}
        func object(_ value:JSONValue?)throws->[String:JSONValue] {guard case .object(let result)=value else {throw fail()};return result}
        func integer(_ value:JSONValue?,minimum:Int=0,maximum:Int=10000)throws->Int {guard case .integer(let n)=value,n>=minimum,n<=maximum else {throw fail()};return Int(n)}
        func number(_ value:JSONValue?,nonnegative:Bool=true)throws->Double {guard let n=value?.doubleValue,n.isFinite,abs(n)<=OptimizerConfiguration.scalarMaximum,(!nonnegative || n>=0) else {throw fail()};return n}
        func equal(_ actual:Double,_ expected:Double)->Bool {abs(actual-expected)<=max(1e-15,abs(expected)*1e-12)}
        let settings=try object(manifest["settings"]),provenance=try object(manifest["provenance"])
        let checkpoint=try object(manifest["checkpoint"])
        let contract = try checkpoint["contract"].map {try object($0)} ?? [:]
        if let expected,let expectedConfiguration {try require(expected.configuration==expectedConfiguration)}
        guard let configuration=expected?.configuration ?? expectedConfiguration else {
            try require(settings["schedulerRecipe"]==nil && provenance["scheduler"]==nil && contract["scheduler"]==nil && manifest["rateTrace"]==nil && manifest["schedulerState"]==nil && trace==nil)
            return
        }
        guard let opt=optimizer?.configuration ?? optimizerConfiguration else {throw fail()}
        if let expected {try require(expected.source.domain == .control && !expected.source.localID.isEmpty && expected.source.localID.utf8.count<=128)}
        if configuration.componentID == "scheduler_cosine_annealing_lr" {try require(configuration.values["eta_min"]!.doubleValue! <= opt.learningRate)}
        if configuration.componentID == "scheduler_reduce_on_plateau" {try require(configuration.values["min_lr"]!.doubleValue! <= opt.learningRate)}
        if configuration.componentID == "scheduler_one_cycle_lr",configuration.values["cycle_momentum"] == .bool(true) {
            try require(opt.componentID != "optimizer_adagrad")
            if ["optimizer_adam","optimizer_adamw"].contains(opt.componentID) {try require(configuration.values["max_momentum"]!.doubleValue! <= OptimizerConfiguration.betaMaximum)}
            if opt.componentID == "optimizer_sgd",opt.values["nesterov"] == .bool(true) {try require(configuration.values["base_momentum"]!.doubleValue! > 0 && opt.values["dampening"]!.doubleValue! == 0)}
        }
        func matches(_ raw:JSONValue?)throws->Bool {guard let raw,case .object(let envelope)=raw,case .object(let config)=envelope["config"],Set(config.keys)==Set(configuration.values.keys) else {return false};return try SchedulerConfiguration.decodeRecipe(raw)==configuration}
        let names=["scheduler_step_lr":"StepLR","scheduler_cosine_annealing_lr":"CosineAnnealingLR","scheduler_reduce_on_plateau":"ReduceLROnPlateau","scheduler_one_cycle_lr":"OneCycleLR","scheduler_linear_lr":"LinearLR"]
        let scheduler=try object(provenance["scheduler"]),stored=try object(contract["scheduler"]),runtime=try object(manifest["runtime"])
        guard let version=runtime["torch"]?.stringValue,!version.isEmpty,version.utf8.count<=128 else {throw fail()}
        try require(try matches(settings["schedulerRecipe"]))
        try require(Set(scheduler.keys)==["recipe","implementation","torchVersion"] && scheduler["implementation"] == .string(names[configuration.componentID]!) && scheduler["torchVersion"] == .string(version))
        try require(try matches(scheduler["recipe"]))
        try require(Set(stored.keys)==["schemaVersion","recipe"] && stored["schemaVersion"] == .integer(1))
        try require(try matches(stored["recipe"]))
        let metadata=try object(manifest["rateTrace"]),content=try object(trace),timing=configuration.timing
        try require(Set(metadata.keys)==["schemaVersion","path","bytes","sha256","records","timing"] && metadata["schemaVersion"] == .integer(1) && metadata["path"] == .string("rate-trace.json") && metadata["timing"] == .string(timing))
        _=try integer(metadata["bytes"],minimum:1,maximum:4_000_000)
        guard let pin=metadata["sha256"]?.stringValue,WorkflowArchiveCodec.isHash(pin) else {throw fail()}
        try require(Set(content.keys)==["schemaVersion","recipe","timing","initialization","records"] && content["schemaVersion"] == .integer(1) && content["timing"] == .string(timing))
        try require(try matches(content["recipe"]))
        guard case .array(let records)=content["records"] else {throw fail()}
        let epochs=try integer(settings["epochs"],minimum:1),batches=try gradientIdentity?.optimizerUpdatesPerEpoch ?? integer(settings["batchesPerEpoch"],minimum:1,maximum:100_000_000),isBatch=configuration.componentID=="scheduler_one_cycle_lr"
        let (total,overflow)=epochs.multipliedReportingOverflow(by:isBatch ? batches:1)
        try require(!overflow && total<=10000 && records.count==total && records.count==integer(metadata["records"]))
        try require(metrics.count==epochs && manifest["schedulerState"]==nil)
        let ownership=try gradientIdentity?.optimizerOwnership ?? optimizerParameterOwnership(graph,context:context,capturedRegistry:capturedRegistry),c=configuration.values,base=opt.learningRate
        func value(_ key:String)->Double {c[key]!.doubleValue!}
        func point(_ tick:Int)throws->(Double,Double?,Bool) {
            var rate=base,momentum:Double?
            switch configuration.componentID {
            case "scheduler_step_lr":rate=base*pow(value("gamma"),Double(tick/Int(value("step_size"))))
            case "scheduler_cosine_annealing_lr":try require(Double(tick)<=value("T_max"));rate=value("eta_min")+(base-value("eta_min"))*(1+cos(.pi*Double(tick)/value("T_max")))/2
            case "scheduler_linear_lr":rate=base*(value("start_factor")+(value("end_factor")-value("start_factor"))*min(Double(tick),value("total_iters"))/value("total_iters"))
            case "scheduler_one_cycle_lr":
                let n=value("planned_total_batches"),initial=value("max_lr")/value("div_factor"),minimum=initial/value("final_div_factor"),first=value("pct_start")*n-1,three=c["three_phase"] == .bool(true)
                try require(Double(tick)<=n)
                let ends=three ? [first,2*value("pct_start")*n-2,n-1]:[first,n-1]
                let pairs=three ? [(initial,value("max_lr")),(value("max_lr"),initial),(initial,minimum)]:[(initial,value("max_lr")),(value("max_lr"),minimum)]
                let momentums=three ? [(value("max_momentum"),value("base_momentum")),(value("base_momentum"),value("max_momentum")),(value("max_momentum"),value("max_momentum"))]:[(value("max_momentum"),value("base_momentum")),(value("base_momentum"),value("max_momentum"))]
                var start=0.0
                for index in ends.indices {
                    if Double(tick)<=ends[index] || index==ends.count-1 {
                        let fraction=(Double(tick)-start)/(ends[index]-start)
                        func mix(_ pair:(Double,Double))->Double {c["anneal_strategy"] == .string("cos") ? pair.1+(pair.0-pair.1)*(1+cos(.pi*fraction))/2:pair.0+(pair.1-pair.0)*fraction}
                        rate=mix(pairs[index]);if c["cycle_momentum"] == .bool(true) {momentum=mix(momentums[index])};break
                    }
                    start=ends[index]
                }
            default:break
            }
            let unused=isBatch && Double(tick)==value("planned_total_batches")
            try require(rate.isFinite && abs(rate)<=OptimizerConfiguration.scalarMaximum && (rate>=0 || unused))
            if let momentum {try require(momentum.isFinite && (0...1).contains(momentum) && (!["optimizer_adam","optimizer_adamw"].contains(opt.componentID) || momentum<=OptimizerConfiguration.betaMaximum))}
            return (rate,momentum,unused)
        }
        let initialization=try object(content["initialization"]),first=try point(0)
        try require(Set(initialization.keys)==(first.1 == nil ? ["tick","rate"]:["tick","rate","momentum"]) && initialization["tick"] == .integer(0) && equal(number(initialization["rate"]),first.0))
        if let momentum=first.1 {try require(equal(number(initialization["momentum"]),momentum))}
        var plateauRate=base,best:Double?,bad=0,cooldown=0
        for (index,raw) in records.enumerated() {
            let row=try object(raw),tick=index+1,epoch=isBatch ? index/batches+1:tick
            try validateSchedulerRecord(raw,configuration:configuration,expectedTick:tick,parameterCount:ownership.count)
            var keys:Set<String>=["tick","epoch","usedRate","nextRate","nextRateUnused"]
            if isBatch {keys.formUnion(["batch","updatedParameterIndices"]);if c["cycle_momentum"] == .bool(true) {keys.formUnion(["usedMomentum","nextMomentum"])}}
            if configuration.componentID=="scheduler_reduce_on_plateau" {keys.formUnion(["metric","metricValue","changed"])}
            try require(Set(row.keys)==keys && row["tick"] == .integer(Int64(tick)) && row["epoch"] == .integer(Int64(epoch)))
            let used=try number(row["usedRate"]),next=try number(row["nextRate"],nonnegative:false)
            if isBatch {
                try require(row["batch"] == .integer(Int64(index%batches+1)))
                guard case .array(let items)=row["updatedParameterIndices"] else {throw fail()}
                let indices=try items.map {try integer($0,maximum:ownership.count-1)}
                try require(indices==Array(Set(indices)).sorted())
            }
            if configuration.componentID=="scheduler_reduce_on_plateau" {
                let selected=c["metric"]!.stringValue!,metric=try number(row["metricValue"],nonnegative:false),metricKey=selected=="validation_loss" ? "validationLoss":"accuracy"
                try require(row["metric"] == .string(selected) && metric==number(metrics[epoch-1][metricKey],nonnegative:false) && equal(used,plateauRate))
                let minimum=c["mode"] == .string("min"),relative=c["threshold_mode"] == .string("rel"),threshold=value("threshold")
                let good:Bool
                if let best {good=minimum ? metric < (relative ? best*(1-threshold):best-threshold):metric > (relative ? best*(1+threshold):best+threshold)} else {good=true}
                if good {best=metric;bad=0} else {bad+=1}
                if cooldown>0 {cooldown-=1;bad=0}
                if bad>Int(value("patience")) {let candidate=max(plateauRate*value("factor"),value("min_lr"));if plateauRate-candidate>value("eps") {plateauRate=candidate};cooldown=Int(value("cooldown"));bad=0}
                try require(equal(next,plateauRate) && row["changed"] == .bool(next != used) && row["nextRateUnused"] == .bool(false))
            } else {
                let before=try point(index),after=try point(tick)
                try require(equal(used,before.0) && equal(next,after.0) && row["nextRateUnused"] == .bool(after.2))
                if let momentum=before.1 {try require(equal(number(row["usedMomentum"]),momentum))}
                if let momentum=after.1 {try require(equal(number(row["nextMomentum"]),momentum))}
            }
        }
    }

    /// Independent native geometry, in the worker's deterministic module/parameter registration order.
    static func optimizerParameterOwnership(_ graph:GraphDocument,context:GraphValidationContext = .preparedWorkflow,capturedRegistry:ComponentRegistry?=nil,initializationOwnership:Bool=false)throws->[JSONValue] {
        let registry=try capturedRegistry ?? ComponentRegistry.builtIn(),validation=GraphValidator.validate(graph,registry:registry,context:context)
        guard validation.isValid else {throw WorkflowPhaseError("Optimizer ownership requires an executable captured model.")}
        let nodes=Dictionary(uniqueKeysWithValues:graph.nodes.map {($0.id,$0)})
        // Installed Torch2.14.1 ModuleDict attributes determine the reviewed collision policy.
        // Node identities colliding with these attributes use the same SHA256-derived module key.
        let reserved:Set<String>=["T_destination", "__annotations__", "__call__", "__class__", "__contains__", "__delattr__", "__delitem__", "__dict__", "__dir__", "__doc__", "__eq__", "__firstlineno__", "__format__", "__ge__", "__getattr__", "__getattribute__", "__getitem__", "__getstate__", "__gt__", "__hash__", "__init__", "__init_subclass__", "__iter__", "__le__", "__len__", "__lt__", "__module__", "__ne__", "__new__", "__reduce__", "__reduce_ex__", "__repr__", "__setattr__", "__setitem__", "__setstate__", "__sizeof__", "__static_attributes__", "__str__", "__subclasshook__", "__weakref__", "_apply", "_backward_hooks", "_backward_pre_hooks", "_buffers", "_call_impl", "_compiled_call_impl", "_forward_hooks", "_forward_hooks_always_called", "_forward_hooks_with_kwargs", "_forward_pre_hooks", "_forward_pre_hooks_with_kwargs", "_get_backward_hooks", "_get_backward_pre_hooks", "_get_name", "_is_full_backward_hook", "_load_from_state_dict", "_load_state_dict_post_hooks", "_load_state_dict_pre_hooks", "_maybe_warn_non_full_backward_hook", "_modules", "_named_members", "_non_persistent_buffers_set", "_parameters", "_register_load_state_dict_pre_hook", "_register_state_dict_hook", "_replicate_for_data_parallel", "_save_to_state_dict", "_slow_forward", "_state_dict_hooks", "_state_dict_pre_hooks", "_version", "_wrapped_call_impl", "add_module", "apply", "bfloat16", "buffers", "call_super_init", "children", "clear", "compile", "cpu", "cuda", "double", "dump_patches", "eval", "extra_repr", "float", "forward", "get_buffer", "get_extra_state", "get_parameter", "get_submodule", "half", "ipu", "items", "keys", "load_state_dict", "modules", "mtia", "named_buffers", "named_children", "named_modules", "named_parameters", "parameters", "pop", "register_backward_hook", "register_buffer", "register_forward_hook", "register_forward_pre_hook", "register_full_backward_hook", "register_full_backward_pre_hook", "register_load_state_dict_post_hook", "register_load_state_dict_pre_hook", "register_module", "register_parameter", "register_state_dict_post_hook", "register_state_dict_pre_hook", "requires_grad_", "set_extra_state", "set_submodule", "share_memory", "state_dict", "to", "to_empty", "train", "training", "type", "update", "values", "xpu", "zero_grad"]
        var moduleTypes:Set<String>=["rnn","lstm","gru","rnn_cell","lstm_cell","gru_cell","prelu","dropout","linear","lazy_linear","bilinear","layer_norm","rms_norm","embedding","embedding_bag","conv1d","conv2d","conv3d","conv_transpose1d","conv_transpose2d","conv_transpose3d","depthwise_conv2d","batch_norm1d","batch_norm2d","instance_norm1d","instance_norm2d","group_norm"]
        if initializationOwnership || graph.compositionPlan != nil {moduleTypes.formUnion(["multihead_attention","transformer","transformer_encoder","transformer_decoder","scaled_dot_product_attention"])}
        var registered=Set<String>(),result:[JSONValue]=[]
        let composition = try compositionRegistration(graph, registry: registry)
        let compositionPaths = Dictionary(uniqueKeysWithValues: composition)
        let ownershipOrder = graph.compositionPlan == nil ? validation.orderedNodeIDs : composition.map { $0.0 }
        for id in ownershipOrder {
            guard let node=nodes[id],moduleTypes.contains(node.type) else {continue}
            let values=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:graph.schemaVersion,registry:registry)
            var moduleKey=id
            if reserved.contains(moduleKey) || registered.contains(moduleKey) {
                moduleKey="kwip_"+WorkflowPhaseArtifacts.hash(Data(id.utf8))
                while nodes[moduleKey] != nil || reserved.contains(moduleKey) || registered.contains(moduleKey) {moduleKey="_"+moduleKey}
            }
            registered.insert(moduleKey)
            func number(_ key:String)throws->Int {guard let value=values[key]?.integerValue,value>0 else {throw WorkflowPhaseError("Optimizer learned-parameter dimensions are unavailable.")};return value}
            func dimensions(_ key:String)throws->[Int] {
                if let value=values[key]?.integerValue {return [value]}
                guard case .array(let items)=values[key] else {throw WorkflowPhaseError("Optimizer parameter directions are unavailable.")}
                return try items.map {guard let value=$0.integerValue,value>0 else {throw WorkflowPhaseError("Optimizer parameter directions are invalid.")};return value}
            }
            func append(_ field:String,_ shape:[Int]) {
                var value:[String:JSONValue]=["name":.string((compositionPaths[id] ?? ("layers."+moduleKey))+"."+field),"shape":.array(shape.map {.integer(Int64($0))}),"dtype":.string("float32")]
                if initializationOwnership {value["node"] = .string(id);value["parameter"] = .string(field)}
                result.append(.object(value))
            }
            var weight:[Int]?,bias:[Int]?
            switch node.type {
            case "multihead_attention","transformer","transformer_encoder","transformer_decoder","scaled_dot_product_attention","attention_mask":
                for parameter in try AttentionConfiguration(componentID:node.type,values:values).parameterShapes {append(parameter.name,parameter.shape)}
            case "rnn","lstm","gru","rnn_cell","lstm_cell","gru_cell":
                for parameter in try RecurrentConfiguration(componentID:node.type,values:values).parameterShapes {append(parameter.name,parameter.shape)}
            case "dropout":break
            case "linear":weight=try [number("out_features"),number("in_features")];if values["bias"] == .bool(true) {bias=try [number("out_features")]}
            case "lazy_linear":
                guard let edge=graph.edges.first(where:{$0.target == id && $0.targetPort == "in"}),let input=validation.outputTensorContracts[.init(edge.source,edge.sourcePort)]?.layout.sample.dimensions.last else {throw WorkflowPhaseError("Lazy optimizer ownership needs its resolved input width.")}
                weight=try [number("out_features"),input];if values["bias"] == .bool(true) {bias=try [number("out_features")]}
            case "bilinear":weight=try [number("out_features"),number("in1_features"),number("in2_features")];if values["bias"] == .bool(true) {bias=try [number("out_features")]}
            case "embedding", "embedding_bag":if initializationOwnership || values["freeze"] != .bool(true) {weight=try [number("num_embeddings"),number("embedding_dim")]}
            case "prelu":weight=try [number("num_parameters")]
            case "rms_norm":if values["elementwise_affine"] == .bool(true) {weight=try dimensions("normalized_shape")}
            case "layer_norm":if values["elementwise_affine"] == .bool(true) {weight=try dimensions("normalized_shape");if values["bias"] == .bool(true) {bias=weight}}
            case "batch_norm1d","batch_norm2d","instance_norm1d","instance_norm2d","group_norm":
                let configuration=try StatefulNormalizationConfiguration(componentID:node.type,values:values)
                if configuration.affine {weight=[configuration.channelCount];if configuration.bias {bias=weight}}
            case "depthwise_conv2d":
                let configuration=try DepthwiseConv2dConfiguration(values:values).convolution
                weight=[configuration.outputChannels,configuration.inputChannels/configuration.groups]+configuration.kernelSize
                if configuration.bias {bias=[configuration.outputChannels]}
            default:
                let input=try number("in_channels"),output=try number("out_channels"),groups=try number("groups"),kernel=try dimensions("kernel_size")
                weight=(node.type.hasPrefix("conv_transpose") ? [input,output/groups]:[output,input/groups])+kernel
                if values["bias"] == .bool(true) {bias=[output]}
            }
            if let weight {append("weight",weight)}
            if let bias {append("bias",bias)}
        }
        return result
    }

    static func validate(_ manifest:[String:JSONValue],mode:WorkflowTrainingMode,targetEpochs:Int,
                         parent:WorkflowLifecycleParentEvidence?,graph:GraphDocument?=nil,hasScheduler:Bool=false,hasObjective:Bool=false,hasGradientPolicy:Bool=false,hasTrainability:Bool=false,gradientIdentity:WorkflowGradientPolicyIdentity?=nil,dataLoadingIdentity:WorkflowDataLoadingIdentity?=nil)throws {
        func require(_ valid:Bool)throws {
            guard valid else {throw WorkflowPhaseError("Training lifecycle differs from the captured mode, parent checkpoint, epoch range or state-transfer policy.")}
        }
        guard case .object(let lineage)=manifest["lineage"],case .object(let settings)=manifest["settings"] else {
            throw WorkflowPhaseError("Training lifecycle evidence is missing.")
        }
        if let dataLoadingIdentity {
            try WorkflowDataLoadingVerifier.validateAttachment(manifest:manifest,identity:dataLoadingIdentity)
        } else {
            try require(manifest["dataLoading"]==nil && settings["datasetRecipe"]==nil && settings["loaderRecipe"]==nil)
            if case .object(let provenance)=manifest["provenance"] {try require(provenance["dataLoading"]==nil)}
            if case .object(let checkpoint)=manifest["checkpoint"] {
                try require(checkpoint["dataLoadingState"]==nil)
                if case .object(let contract)=checkpoint["contract"] {try require(contract["dataLoading"]==nil)}
            }
        }
        let hasDataLoading=dataLoadingIdentity != nil
        let loadingState=hasDataLoading ? ["dataLoading"]:[]
        try require((1...10_000).contains(targetEpochs))
        let hasBuffers = try graph?.nodes.contains { node in
            guard StatefulNormalizationConfiguration.componentIDs.contains(node.type) else { return false }
            return try StatefulNormalizationConfiguration(componentID:node.type,values:node.config).tracksRunningStatistics
        } ?? false
        let modelState = hasBuffers ? ["parameters","buffers"] : ["parameters"]
        let scheduleState=hasScheduler ? ["scheduler","rateTrace"] : []
        let objectiveState=hasObjective ? ["objective","objectiveBuffers"] : []
        let policyState=(hasGradientPolicy ? ["gradientPolicy"]:[])+(hasTrainability ? ["trainability"]:[])
        if hasGradientPolicy || hasTrainability {
            try require(gradientIdentity != nil && (gradientIdentity?.gradientPolicy != nil) == hasGradientPolicy && (gradientIdentity?.trainability != nil) == hasTrainability)
        }
        let all=modelState+["optimizer","randomGenerator","epoch","metrics"]+(mode == .fresh ? scheduleState+policyState+loadingState+objectiveState : objectiveState+scheduleState+policyState+loadingState)
        let isResume=mode == .exactResume
        let workerMode=mode == .fresh ? "newTraining" : isResume ? "exactResume" : "weightsOnlyFineTune"
        let restored=mode == .fresh ? [] : isResume ? all : modelState
        var continuationReset=["optimizer","randomGenerator","epoch","metrics"]
        if parent?.hasObjective == true {continuationReset += ["objective","objectiveBuffers"]}
        if parent?.hasScheduler == true {continuationReset += ["scheduler","rateTrace"]}
        if parent?.hasGradientPolicy == true {continuationReset.append("gradientPolicy")}
        if parent?.hasTrainability == true {continuationReset.append("trainability")}
        if parent?.hasDataLoading == true {continuationReset.append("dataLoading")}
        if hasScheduler && parent?.hasScheduler != true {continuationReset += ["scheduler","rateTrace"]}
        if hasGradientPolicy && parent?.hasGradientPolicy != true {continuationReset.append("gradientPolicy")}
        if hasTrainability && parent?.hasTrainability != true {continuationReset.append("trainability")}
        if hasDataLoading && parent?.hasDataLoading != true {continuationReset.append("dataLoading")}
        if hasObjective && parent?.hasObjective != true {continuationReset += ["objective","objectiveBuffers"]}
        let reset=mode == .fresh ? all : isResume ? [] : continuationReset
        try require(lineage["mode"] == .string(workerMode)
            && lineage["restoredState"] == .array(restored.map(JSONValue.string))
            && lineage["resetState"] == .array(reset.map(JSONValue.string)))
        var keys:Set<String>=["mode","parentRunID","restoredState","resetState"]
        let initial:Int
        if mode == .fresh {
            try require(parent == nil && lineage["parentRunID"] == .null)
            initial=0
        } else {
            guard let parent else {throw WorkflowPhaseError("Continuation requires verified parent lifecycle evidence.")}
            if isResume {
                try require(parent.hasScheduler == hasScheduler && parent.hasObjective == hasObjective && parent.hasGradientPolicy == hasGradientPolicy && parent.hasTrainability == hasTrainability)
                try require(parent.hasDataLoading==hasDataLoading && parent.dataLoadingIdentity==dataLoadingIdentity)
                if hasDataLoading {guard case .object(let runtime)=manifest["runtime"] else {throw WorkflowPhaseError("Exact data loading continuation requires the actual CPU runtime.")};try require(runtime["device"] == .string("cpu"))}
                if hasGradientPolicy || hasTrainability {
                    guard let selected=gradientIdentity,let stored=parent.gradientIdentity else {throw WorkflowPhaseError("Exact continuation requires verified parent gradient policy and ownership.")}
                    try require(stored.gradientPolicy == selected.gradientPolicy && stored.trainability == selected.trainability && (stored.optimizerOwnership.isEmpty || stored.optimizerOwnership == selected.optimizerOwnership) && stored.microbatchesPerEpoch == selected.microbatchesPerEpoch && stored.optimizerUpdatesPerEpoch == selected.optimizerUpdatesPerEpoch)
                    if hasTrainability {try require(stored.fullOwnership == selected.fullOwnership)}
                    guard case .object(let runtime)=manifest["runtime"] else {throw WorkflowPhaseError("Exact policy continuation requires CPU.")}
                    try require(runtime["device"] == .string("cpu"))
                }
                if hasObjective {guard case .object(let runtime)=manifest["runtime"] else {throw WorkflowPhaseError("Exact objective resume requires the actual CPU runtime.")};try require(runtime["device"] == .string("cpu"))}
            }
            try require(UUID(uuidString:parent.workerRunID) != nil && WorkflowArchiveCodec.isHash(parent.checkpointSHA256)
                && (1...10_000).contains(parent.completedEpoch))
            keys.formUnion(["parentCheckpointSHA256","parentCompletedEpoch","guarantee"])
            try require(lineage["parentRunID"] == .string(parent.workerRunID)
                && lineage["parentCheckpointSHA256"] == .string(parent.checkpointSHA256)
                && lineage["parentCompletedEpoch"] == .integer(Int64(parent.completedEpoch))
                && lineage["guarantee"] == .string(isResume ? "epochBoundaryExactCPU":"weightsOnly"))
            initial=isResume ? parent.completedEpoch : 0
        }
        try require(Set(lineage.keys) == keys && targetEpochs>initial
            && settings["initialEpoch"] == .integer(Int64(initial))
            && settings["epochsExecuted"] == .integer(Int64(targetEpochs-initial)))
    }
}
