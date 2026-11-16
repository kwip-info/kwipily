import Foundation

public enum CapabilityContractValidator {
    private static func issue(_ code:String,_ path:String,_ message:String)->ContractIssue {ContractIssue(code,path:path,message:message)}
    private static func versionValid(_ version:ContractVersion)->Bool {version.major >= 0 && version.minor >= 0 && version.patch >= 0}
    private static func named(_ value:String)->Bool {!value.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty}
    private static func runtimePinned(_ value:String)->Bool {value.range(of:#"^[0-9]+\.[0-9]+\.[0-9]+([.+-][A-Za-z0-9._-]+)?$"#,options:.regularExpression) != nil}
    private static func digestValid(_ value:String)->Bool {value.count == 64 && value.allSatisfy {"0123456789abcdef".contains($0)}}

    public static func validate(_ package:CapabilityPackageDescriptor)->ContractValidation {
        var issues:[ContractIssue]=[]
        if package.schemaVersion != 1 {issues.append(issue("schema.version","package","Unsupported package schema; no implicit migration is permitted."))}
        if !named(package.id) || !versionValid(package.version) {issues.append(issue("package.identity","package","Package identity and nonnegative version are required."))}
        if !package.bundled {issues.append(issue("package.trust","package","External executable packages are not enabled by this host contract."))}
        if Set(package.capabilities.map(\.id)).count != package.capabilities.count {issues.append(issue("capability.duplicate","package","Capability IDs must be unique within a package."))}
        for capability in package.capabilities {
            if capability.packageID != package.id {issues.append(issue("package.membership",capability.id,"Capability must identify its containing package."))}
            if capability.backend != .portable && !(package.backendVersions[capability.backend.rawValue].map(runtimePinned) ?? false) {issues.append(issue("runtime.unpinned",capability.id,"Record an exact framework runtime version for this capability."))}
            issues += validate(capability).issues
        }
        return ContractValidation(issues)
    }

    public static func validate(_ capability:CapabilityDescriptor)->ContractValidation {
        let path=capability.id;var issues:[ContractIssue]=[]
        if !named(capability.id) || !named(capability.packageID) || !versionValid(capability.version) {issues.append(issue("capability.identity",path,"Stable capability/package IDs and a nonnegative version are required."))}
        if capability.lifecycle.phases.isEmpty || Set(capability.lifecycle.phases).count != capability.lifecycle.phases.count {issues.append(issue("lifecycle.phases",path,"Declare distinct supported execution phases."))}
        if capability.devices.isEmpty || capability.devices.contains(.runtimeSelected) {issues.append(issue("device.support",path,"Declare concrete supported devices; runtime selection is a port constraint only."))}
        if capability.availability == .existingPOCAdapter && !(capability.adapterID.map(named) ?? false) {issues.append(issue("adapter.missing",path,"Existing adapter availability needs an explicit host adapter identity."))}
        let keys=capability.fields.map(\.key)
        if keys.contains(where:{!named($0)}) || Set(keys).count != keys.count {issues.append(issue("configuration.keys",path,"Configuration keys must be nonempty and unique."))}
        for field in capability.fields where !field.isValidDefinition {
            issues.append(issue("configuration.bounds",path+"."+field.key,"Configuration defaults and constraints must match the declared field type."))
        }
        let identities=capability.ports.map { $0.direction.rawValue+":"+$0.id }
        if Set(identities).count != identities.count {issues.append(issue("port.duplicate",path,"Port IDs must be unique within each direction."))}
        if Set(capability.lifecycle.state.map(\.key)).count != capability.lifecycle.state.count {issues.append(issue("state.duplicate",path,"State keys must be unique."))}
        for state in capability.lifecycle.state {
            if !named(state.key) || !capability.lifecycle.phases.contains(state.resetPhase) {issues.append(issue("state.reset",path+"."+state.key,"Owned state needs a name and supported explicit reset phase."))}
        }
        // A fixed objective recipe is a control value consumed during scoring, not a fit/train operator.
        // Actual configured buffers belong to execution objectiveState; the recipe owns no learned tensors.
        let fixedObjectiveRecipe = capability.ports.count == 1
            && capability.ports[0].id == "out" && capability.ports[0].direction == .output
            && capability.ports[0].schema.kind == .state
            && capability.ports[0].schema.state == StatePortContract(key:"objectiveRecipe",kind:.objective)
            && capability.lifecycle.state.count == 1
            && capability.lifecycle.state[0].key == "objectiveRecipe" && capability.lifecycle.state[0].kind == .objective
            && capability.lifecycle.state[0].sharing == .privateInstance
            && capability.lifecycle.resume == .none
            && Set(capability.lifecycle.phases) == Set([ContractPhase.initialize,.train,.evaluate,.checkpoint,.resume,.export,.reset])
        // A starting-value recipe owns only its control value. The captured model owns its learned arrays.
        let fixedInitializationRecipe = capability.fields.isEmpty && capability.ports.count == 1
            && capability.ports[0].id == "out" && capability.ports[0].direction == .output
            && capability.ports[0].schema.kind == .state
            && capability.ports[0].schema.state == StatePortContract(key:"initializationRecipe",kind:.initialization)
            && Set(capability.ports[0].phases) == Set([ContractPhase.initialize,.train])
            && capability.lifecycle.state == [StateDeclaration("initializationRecipe",kind:.initialization)]
            && capability.lifecycle.resume == .none
            && Set(capability.lifecycle.phases) == Set([ContractPhase.initialize,.train,.checkpoint,.export,.reset])
        // These policies own only a serialized control recipe. Model arrays and
        // optimizer ownership stay with Train; neither control fits observations.
        let policyKey: String? = capability.id == "gradient_policy" ? "gradientPolicyRecipe" : capability.id == "trainability" ? "trainabilityRecipe" : nil
        let policyKind: StateKind = capability.id == "gradient_policy" ? .gradientPolicy : .trainability
        let fixedTrainingPolicy = policyKey.map { key in
            capability.ports == [PortContract("out", direction: .output,
                schema: .state(StatePortContract(key: key, kind: policyKind)), phases: [.initialize, .train, .evaluate])]
            && capability.lifecycle.state == [StateDeclaration(key, kind: policyKind)]
            && capability.lifecycle.resume == .none
            && Set(capability.lifecycle.phases) == Set([ContractPhase.initialize, .train, .evaluate, .checkpoint, .resume, .export, .reset])
        } ?? false
        if capability.lifecycle.phases.contains(.fit) || capability.lifecycle.phases.contains(.train) {
            if capability.lifecycle.fitBoundary != .trainingPartitionOnly && !((fixedObjectiveRecipe || fixedInitializationRecipe || fixedTrainingPolicy) && capability.lifecycle.fitBoundary == .notApplicable) {issues.append(issue("fit.boundary",path,"Fit/train operations require an explicit training-partition-only boundary."))}
        }
        if capability.lifecycle.resume == .fullTraining {
            let available=Set(capability.lifecycle.state.filter(\.serializable).map(\.kind))
            if !fullResumeState.isSubset(of:available) || !capability.lifecycle.phases.contains(.resume) {issues.append(issue("resume.state",path,"Full training resume needs resume phase and serializable parameters, optimizer, RNG and data position."))}
        }
        if let boundary=capability.boundary {
            if boundary.source == boundary.destination || boundary.destination != capability.backend || !boundary.detached || !named(boundary.adapterID) {issues.append(issue("backend.boundary",path,"Conversion must name an adapter, distinct backends and an explicitly detached destination."))}
            if capability.ports.contains(where:{$0.schema.tensor?.gradients == true || $0.schema.kind == .state || $0.schema.kind == .estimator}) {issues.append(issue("backend.state",path,"A backend conversion cannot imply gradient, estimator or live-state continuity."))}
        }
        for port in capability.ports {
            let portPath=path+"."+port.direction.rawValue+"."+port.id
            if !named(port.id) || port.phases.isEmpty || !Set(port.phases).isSubset(of:Set(capability.lifecycle.phases)) {issues.append(issue("port.phase",portPath,"Port needs a stable ID and phases supported by its component."))}
            if let count=port.maximumConnections,count < 1 {issues.append(issue("port.multiplicity",portPath,"Maximum connections must be positive, or nil for unbounded."))}
            issues += schemaIssues(port.schema,path:portPath,fields:Set(capability.fields.filter{$0.kind == .integer}.map(\.key)))
            if let tensor=port.schema.tensor ?? port.schema.dataset?.tensor {
                let expected=port.direction == .input ? capability.boundary?.source ?? capability.backend : capability.backend
                if tensor.backend != expected {issues.append(issue("backend.adapter",portPath,"Tensor backend differs from its execution adapter; declare a conversion boundary."))}
                if tensor.device != .runtimeSelected && tensor.device != .host && !capability.devices.contains(tensor.device) {issues.append(issue("device.unsupported",portPath,"Port device is absent from the adapter's declared support."))}
            }
            if let estimator=port.schema.estimator,estimator.backend != capability.backend {issues.append(issue("estimator.backend",portPath,"Estimator state belongs to its declared backend adapter."))}
            if let state=port.schema.state {
                guard let declaration=capability.lifecycle.state.first(where:{$0.key == state.key}) else {issues.append(issue("state.undeclared",portPath,"State port requires an owned lifecycle declaration."));continue}
                if state.kind != declaration.kind || state.sharing != declaration.sharing {issues.append(issue("state.contract",portPath,"State port kind and sharing must match its declaration."))}
            }
        }
        if !capability.learning.challengeIDs.isEmpty && capability.learning.assessedOutputs.isEmpty {issues.append(issue("learning.evidence",path,"Output-assessed challenges must identify their measured outputs."))}
        return ContractValidation(issues)
    }

    private static func schemaIssues(_ schema:PortSchema,path:String,fields:Set<String>)->[ContractIssue] {
        var issues:[ContractIssue]=[]
        let present:[SchemaKind]=[(SchemaKind.tensor,schema.tensor != nil),(.table,schema.table != nil),(.dataset,schema.dataset != nil),(.estimator,schema.estimator != nil),(.state,schema.state != nil),(.artifact,schema.artifact != nil)].filter {$0.1}.map {$0.0}
        guard present == [schema.kind] else {return [issue("schema.payload",path,"Exactly one schema payload must match the declared kind.")]}
        if let tensor=schema.tensor {
            if let axes=tensor.leadingAxes, axes.minimum < 0 || axes.maximum < axes.minimum || axes.maximum > 8 || axes.minimum+tensor.dimensions.count < 1 || axes.maximum+tensor.dimensions.count > 8 {
                issues.append(issue("shape.rankRange",path,"Variable leading axes must describe a bounded total tensor rank from 1 through 8."))
            }
            if tensor.dtype == .string {issues.append(issue("tensor.dtype",path,"String columns require an explicit encoding adapter before tensor conversion."))}
            for (axis,dimension) in tensor.dimensions.enumerated() {
                switch dimension.kind {
                case .fixed: if (dimension.value ?? 0) <= 0 || dimension.name != nil {issues.append(issue("shape.dimension",path+".\(axis)","Fixed dimensions must be positive and unnamed."))}
                case .symbol: if !(dimension.name.map(named) ?? false) || dimension.value != nil {issues.append(issue("shape.symbol",path+".\(axis)","Symbolic dimensions require only a nonempty name."))}
                case .parameter: if !fields.contains(dimension.name ?? "") || dimension.value != nil {issues.append(issue("shape.parameter",path+".\(axis)","Parameter dimensions must refer to a declared configuration field."))}
                }
            }
        }
        if let table=schema.table {
            if Set(table.columns.map(\.name)).count != table.columns.count || table.columns.contains(where:{!named($0.name)}) {issues.append(issue("table.columns",path,"Table column names must be nonempty and unique."))}
        }
        if let dataset=schema.dataset {
            if (dataset.table == nil) == (dataset.tensor == nil) {issues.append(issue("dataset.element",path,"Dataset must declare exactly one table or tensor element schema."))}
            if let table=dataset.table {issues += schemaIssues(.table(table),path:path+".element",fields:fields)}
            if let tensor=dataset.tensor {issues += schemaIssues(.tensor(tensor),path:path+".element",fields:fields)}
        }
        if let estimator=schema.estimator,!named(estimator.family) || estimator.backend == .portable {issues.append(issue("estimator.identity",path,"An estimator requires a framework backend and family identity."))}
        if let artifact=schema.artifact,!named(artifact.format) {issues.append(issue("artifact.format",path,"Artifacts require an explicit serialization format."))}
        return issues
    }

    public static func connection(from:ContractEndpoint,to:ContractEndpoint,phase:ContractPhase,existingInputConnections:Int=0)->ContractValidation {
        var issues=validate(from.capability).issues+validate(to.capability).issues
        guard let source=from.capability.ports.first(where:{$0.id == from.portID && $0.direction == .output}),let target=to.capability.ports.first(where:{$0.id == to.portID && $0.direction == .input}) else {return ContractValidation(issues+[issue("connection.port","connection","Connect an existing output port to an existing input port.")])}
        if !named(from.instanceID) || !named(to.instanceID) {issues.append(issue("connection.instance","connection","Connections require explicit instance identities."))}
        if !source.phases.contains(phase) || !target.phases.contains(phase) {issues.append(issue("connection.phase","connection","Both ports must participate in the requested lifecycle phase."))}
        if existingInputConnections < 0 || target.maximumConnections.map({existingInputConnections >= $0}) == true {issues.append(issue("connection.multiplicity","connection","Target port has reached its connection limit."))}
        guard issues.isEmpty else{return ContractValidation(issues)}
        issues += compatible(source.schema,target.schema,from:from,to:to,path:"connection")
        return ContractValidation(issues)
    }
    private static func compatible(_ source:PortSchema,_ target:PortSchema,from:ContractEndpoint,to:ContractEndpoint,path:String)->[ContractIssue] {
        guard source.kind == target.kind else{return [issue("connection.kind",path,"Different payload kinds need an explicit data conversion component.")]}
        switch source.kind {
        case .tensor:
            guard let a=source.tensor,let b=target.tensor else{return [issue("schema.payload",path,"Missing tensor schema.")]}
            var issues:[ContractIssue]=[]
            if a.backend != b.backend {issues.append(issue("connection.backend",path,"Cross-backend tensors require an explicit detached conversion component."))}
            if a.dtype != b.dtype {issues.append(issue("connection.dtype",path,"Different dtypes require an explicit cast."))}
            if a.device != .runtimeSelected && b.device != .runtimeSelected && a.device != b.device {issues.append(issue("connection.device",path,"Different devices require an explicit transfer."))}
            if b.gradients && !a.gradients {issues.append(issue("connection.gradients",path,"The consumer requires gradient continuity that the producer does not provide."))}
            if a.leadingAxes != nil || b.leadingAxes != nil {
                issues.append(issue("shape.unresolved",path,"Specialize variable-rank ports with the component shape rule before connecting."))
            } else {issues += dimensions(a.dimensions,b.dimensions,from:from,to:to,path:path)}
            return issues
        case .table:
            guard let a=source.table,let b=target.table else{return [issue("schema.payload",path,"Missing table schema.")]}
            var issues:[ContractIssue]=[]
            for column in b.columns {
                guard let supplied=a.columns.first(where:{$0.name == column.name}) else {issues.append(issue("table.missing",path+"."+column.name,"Required column is absent."));continue}
                if supplied.dtype != column.dtype || (supplied.nullable && !column.nullable) {issues.append(issue("table.column",path+"."+column.name,"Column dtype or nullability is incompatible; use an explicit transformation."))}
            }
            if !b.allowsAdditionalColumns && Set(a.columns.map(\.name)) != Set(b.columns.map(\.name)) {issues.append(issue("table.extra",path,"Target uses a closed column schema."))}
            return issues
        case .dataset:
            guard let a=source.dataset,let b=target.dataset else{return [issue("schema.payload",path,"Missing dataset schema.")]}
            var issues:[ContractIssue]=[]
            if b.partition != .unassigned && a.partition != b.partition {issues.append(issue("dataset.partition",path,"Dataset partition does not satisfy the consumer's declared role."))}
            if let at=a.table,let bt=b.table {issues += compatible(.table(at),.table(bt),from:from,to:to,path:path+".element")}
            else if let at=a.tensor,let bt=b.tensor {issues += compatible(.tensor(at),.tensor(bt),from:from,to:to,path:path+".element")}
            else {issues.append(issue("dataset.element",path,"Dataset element kinds do not match."))}
            return issues
        case .estimator:
            guard let a=source.estimator,let b=target.estimator,a.backend == b.backend,a.family == b.family,!b.requiresFitted || a.requiresFitted else{return [issue("estimator.lifecycle",path,"Estimator backend, family and fitted state must satisfy the consumer.")]};return []
        case .state:
            guard let a=source.state,let b=target.state,a == b,let owner=from.stateOwnerID,!owner.isEmpty,owner == to.stateOwnerID else{return [issue("state.owner",path,"State requires matching explicit owner identity, key, kind and sharing.")]}
            if a.sharing == .privateInstance && (owner != from.instanceID || owner != to.instanceID) {return [issue("state.private",path,"Private instance state cannot be wired into another component instance.")]}
            if from.capability.backend != to.capability.backend {return [issue("state.backend",path,"Live state cannot cross backend ownership.")]};return []
        case .artifact:
            guard let a=source.artifact,let b=target.artifact,a.kind == b.kind,a.backend == b.backend,a.format == b.format,resumeRank(a.resume) >= resumeRank(b.resume) else{return [issue("artifact.contract",path,"Artifact kind, backend, format and resume guarantee must match.")]};return []
        }
    }
    private enum DimensionValue {case fixed(Int),variable(String)}
    private static func dimensions(_ source:[DimensionContract],_ target:[DimensionContract],from:ContractEndpoint,to:ContractEndpoint,path:String)->[ContractIssue] {
        guard source.count == target.count else{return [issue("shape.rank",path,"Tensor ranks differ.")]}
        var issues:[ContractIssue]=[],parent:[String:String]=[:],known:[String:Int]=[:]
        func root(_ key:String)->String {var result=key;while let next=parent[result],next != result {result=next};return result}
        func resolve(_ dimension:DimensionContract,_ endpoint:ContractEndpoint,_ side:String,_ axis:Int)->DimensionValue? {
            switch dimension.kind {
            case .fixed:return dimension.value.map { .fixed($0) }
            case .symbol:return .variable(side+":"+(dimension.name ?? ""))
            case .parameter:
                guard let key=dimension.name,let value=endpoint.configuration[key],let number=value.integerValue,number > 0,let field=endpoint.capability.fields.first(where:{$0.key == key}),field.kind == .integer,field.accepts(value) else {issues.append(issue("shape.configuration",path+".\(axis)","A shape parameter is missing, nonintegral or outside its declared bounds."));return nil}
                return .fixed(number)
            }
        }
        func bind(_ key:String,_ value:Int,_ axis:Int) {
            let r=root(key)
            if let prior=known[r],prior != value {issues.append(issue("shape.symbol-conflict",path+".\(axis)","A repeated dimension symbol resolves to incompatible sizes."))}
            known[r]=value
        }
        var variables:Set<String>=[]
        for axis in source.indices {
            guard let a=resolve(source[axis],from,"source",axis),let b=resolve(target[axis],to,"target",axis) else{continue}
            switch (a,b) {
            case (.fixed(let x),.fixed(let y)):if x != y {issues.append(issue("shape.width",path+".\(axis)","Dimension \(x) cannot connect to \(y)."))}
            case (.variable(let key),.fixed(let value)),(.fixed(let value),.variable(let key)):variables.insert(key);bind(key,value,axis)
            case (.variable(let x),.variable(let y)):
                variables.formUnion([x,y]);let xr=root(x),yr=root(y)
                if xr != yr {let old=known[yr];parent[yr]=xr;if let old {bind(xr,old,axis)}}
            }
        }
        if variables.contains(where:{known[root($0)] == nil}) {issues.append(ContractIssue("shape.runtime",path:path,message:"Symbolic dimensions require validation against actual runtime tensors.",severity:.deferred))}
        return issues
    }
    public static func fit(_ capability:CapabilityDescriptor,phase:ContractPhase,partition:PartitionEvidence,heldOutObservationIDs:Set<String>)->ContractValidation {
        var issues=validate(capability).issues
        if ![ContractPhase.fit,.train].contains(phase) || !capability.lifecycle.phases.contains(phase) || capability.lifecycle.fitBoundary != .trainingPartitionOnly {issues.append(issue("fit.phase",capability.id,"This operation is not a declared train-only fit/train phase."))}
        if partition.role != .training {issues.append(issue("fit.partition",capability.id,"Learned preprocessing and training may fit on the training partition only."))}
        if !digestValid(partition.sourceDigest) || !digestValid(partition.membershipDigest) || partition.observationIDs.isEmpty || Set(partition.observationIDs).count != partition.observationIDs.count || partition.observationIDs.contains(where:{!named($0)}) {issues.append(issue("fit.provenance",capability.id,"Source/membership SHA-256 identities and distinct observation IDs are required."))}
        if partition.membershipDigest != PartitionEvidence.membershipDigest(for:partition.observationIDs) {issues.append(issue("fit.membership-digest",capability.id,"Membership digest does not match the declared observation IDs."))}
        if !Set(partition.observationIDs).isDisjoint(with:heldOutObservationIDs) {issues.append(issue("fit.leakage",capability.id,"Training observations overlap held-out membership."))}
        return ContractValidation(issues)
    }
    private static let fullResumeState:Set<StateKind>=[.parameters,.optimizer,.randomGenerator,.dataCursor]
    private static func resumeRank(_ guarantee:ResumeGuarantee)->Int {switch guarantee {case .none:return 0;case .weightsOnly:return 1;case .fullTraining:return 2}}
    public static func artifact(_ artifact:ArtifactIdentity,requiredResume:ResumeGuarantee = .none)->ContractValidation {
        var issues:[ContractIssue]=[]
        if artifact.schemaVersion != 1 || !named(artifact.id) || !named(artifact.producerCapabilityID) || !versionValid(artifact.producerVersion) || !named(artifact.contract.format) {issues.append(issue("artifact.identity",artifact.id,"Artifact schema, producer/version and format must be explicit."))}
        if !digestValid(artifact.contentDigest) || !digestValid(artifact.graphDigest) {issues.append(issue("artifact.digest",artifact.id,"Artifact content and graph require lowercase SHA-256 digests, not paths."))}
        if resumeRank(artifact.contract.resume) < resumeRank(requiredResume) {issues.append(issue("artifact.resume",artifact.id,"This artifact does not promise the requested resume level."))}
        if artifact.contract.resume == .fullTraining && (artifact.contract.kind != .trainingCheckpoint || !fullResumeState.isSubset(of:Set(artifact.includedState))) {issues.append(issue("artifact.resume-state",artifact.id,"Full resume requires a training checkpoint with parameters, optimizer, RNG and data position; weights alone are insufficient."))}
        if artifact.contract.resume == .weightsOnly && !artifact.includedState.contains(.parameters) {issues.append(issue("artifact.weights",artifact.id,"Weight initialization requires parameter state."))}
        return ContractValidation(issues)
    }
}
