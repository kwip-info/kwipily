import Foundation
import KwipMLCore

/// Shared CLI boundary for workflow phases, independent of experiment membership.
enum WorkflowPhaseArguments {
    static func validate(_ args: [String], workspace: URL, projectID: String,
                         directoryID: String, phase: WorkflowLaunchIntent.Phase) throws {
        guard args.count <= 128,args.allSatisfy({!$0.isEmpty && !$0.contains("\0")}) else {throw WorkflowFileLeaseError.unsafePath}
        let preparation:Set<String>=["--dataflow-live-previews","--dataflow-measurements","--defer-database-writes","--destination-bindings","--retain-partial-dataflow","--execute-dataflow","--source-bindings","--dataflow-required-nodes","--dataflow-model-input-nodes","--dataflow-capture-outputs","--dataflow-cache-policy","--dataflow-cache-limit-bytes","--output","--dataflow-reuse-manifest","--dataflow-reuse-manifest-sha256"]
        let training:Set<String>=["--named-input-plan-json","--graph","--output","--registry","--epochs","--seed","--device","--batch-size","--learning-rate","--prepared-dataflow","--prepared-manifest-sha256","--prepared-graph-digest","--prepared-training-node","--prepared-training-port","--prepared-validation-node","--prepared-validation-port","--feature-columns-json","--bag-input-mapping-json","--target-column","--gradient-every-epochs","--gradient-max-batches-per-epoch","--gradient-max-captures","--resume-checkpoint","--finetune-checkpoint","--composite-source","--composite-source-sha256","--capture-gradients","--training-control-json","--training-progress-json","--sample-capture-json","--optimizer-recipe-json", "--scheduler-recipe-json","--initialization-recipe-json","--objective-recipe-json","--objective-target-mapping-json","--objective-scalarization","--gradient-policy-json","--trainability-json","--derivative-capture-json","--dataset-recipe-json","--loader-recipe-json"]
        let allowed=phase == .preparation ? preparation:training
        var values:[String:String]=[:],index=0
        while index<args.count {
            let key=args[index]
            guard allowed.contains(key),values[key] == nil else {throw WorkflowFileLeaseError.unsafePath}
            if key == "--capture-gradients" || key == "--retain-partial-dataflow" {values[key]="true";index+=1;continue}
            guard index+1<args.count,!args[index+1].hasPrefix("--") else {throw WorkflowFileLeaseError.unsafePath}
            let maximum=["--dataset-recipe-json","--loader-recipe-json"].contains(key) ? 32768:key == "--derivative-capture-json" ? DerivativeCapturePolicy.maximumWireBytes:["--gradient-policy-json","--trainability-json"].contains(key) ? GradientPolicyConfiguration.maximumRecipeBytes:key == "--objective-target-mapping-json" ? 524288:key == "--objective-recipe-json" ? LossConfiguration.maximumRecipeBytes:key == "--initialization-recipe-json" ? InitializationConfiguration.maximumRecipeBytes:65536
            guard args[index+1].utf8.count<=maximum else {throw WorkflowFileLeaseError.unsafePath}
            values[key]=args[index+1];index+=2
        }
        if let raw=values["--named-input-plan-json"] {
            guard phase == .training,let bytes=raw.data(using:.utf8),bytes.count<=65536 else {throw WorkflowFileLeaseError.unsafePath}
            guard case .object(let plan)=try JSONDecoder().decode(JSONValue.self,from:bytes) else {throw WorkflowFileLeaseError.unsafePath};_=try NamedInputPlan(values:plan)
        }
        let objectiveKeys:Set<String>=["--objective-recipe-json","--objective-target-mapping-json","--objective-scalarization"]
        let hasObjective=values.keys.contains {objectiveKeys.contains($0)}
        guard args.reduce(0,{$0+$1.utf8.count}) <= (hasObjective ? 786432:131072) else {throw WorkflowFileLeaseError.unsafePath}
        var objectiveReduction = "mean"
        if hasObjective {
            guard phase == .training,let recipe=values["--objective-recipe-json"],let mapping=values["--objective-target-mapping-json"],
                  let scalar=values["--objective-scalarization"],WorkflowObjectiveScalarization(rawValue:scalar) != nil,
                  values["--feature-columns-json"] == nil,values["--target-column"] == nil,
                  let manifest=values["--prepared-dataflow"],let manifestHash=values["--prepared-manifest-sha256"],
                  let graphDigest=values["--prepared-graph-digest"],let trainingNode=values["--prepared-training-node"],
                  let trainingPort=values["--prepared-training-port"],let validationNode=values["--prepared-validation-node"],
                  let validationPort=values["--prepared-validation-port"] else {throw WorkflowFileLeaseError.unsafePath}
            let raw=try JSONDecoder().decode(JSONValue.self,from:Data(recipe.utf8))
            let configuration=try LossConfiguration.decodeRecipe(raw)
            objectiveReduction = configuration.values["reduction"]?.stringValue ?? "mean"
            guard try WorkflowPhaseArtifacts.encode(raw) == WorkflowPhaseArtifacts.encode(configuration.recipeValue) else {throw WorkflowFileLeaseError.unsafePath}
            let target=try JSONDecoder().decode(JSONValue.self,from:Data(mapping.utf8))
            let typed=try WorkflowControlContracts.lossTargetMapping(["lossTargetMapping":target],address:.init(domain:.control,localID:"phase-objective"))
            try configuration.validate(targetMapping:typed)
            try PreparedDataReference(manifestPath:manifest,manifestSHA256:manifestHash,graphDigest:graphDigest,
                trainingNode:trainingNode,trainingPort:trainingPort,validationNode:validationNode,validationPort:validationPort).validate()
            // This is phase admission only. The independently bounded operation envelope
            // and immutable preparation evidence must still pass their own checks.
        }
        if let raw = values["--bag-input-mapping-json"] {
            guard phase == .training, raw.utf8.count <= 32768,
                  values["--feature-columns-json"] == nil,
                  let manifest = values["--prepared-dataflow"], let manifestHash = values["--prepared-manifest-sha256"],
                  let graphDigest = values["--prepared-graph-digest"], let trainingNode = values["--prepared-training-node"],
                  let trainingPort = values["--prepared-training-port"], let validationNode = values["--prepared-validation-node"],
                  let validationPort = values["--prepared-validation-port"] else { throw WorkflowFileLeaseError.unsafePath }
            let mapping = try BagInputMapping(value: JSONDecoder().decode(JSONValue.self, from: Data(raw.utf8)))
            guard try WorkflowPhaseArtifacts.encode(mapping.value) == Data(raw.utf8) else { throw WorkflowFileLeaseError.unsafePath }
            try PreparedDataReference(manifestPath: manifest, manifestSHA256: manifestHash, graphDigest: graphDigest,
                trainingNode: trainingNode, trainingPort: trainingPort, validationNode: validationNode, validationPort: validationPort).validate()
            let packetColumns = Set(mapping.indicesColumns + (mapping.weightColumns ?? []) + (mapping.lengthColumn.map { [$0] } ?? []))
            if hasObjective {
                guard let rawTarget = values["--objective-target-mapping-json"] else { throw WorkflowFileLeaseError.unsafePath }
                let target = try WorkflowControlContracts.lossTargetMapping(["lossTargetMapping": JSONDecoder().decode(JSONValue.self, from: Data(rawTarget.utf8))], address: .init(domain: .control, localID: "phase-bags"))
                guard target.inputColumns == mapping.indicesColumns, packetColumns.isDisjoint(with: target.roles.flatMap(\.columns)) else { throw WorkflowFileLeaseError.unsafePath }
            } else {
                guard let target = values["--target-column"], !target.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                      target.count <= 200, !packetColumns.contains(target) else { throw WorkflowFileLeaseError.unsafePath }
            }
        }
        if let raw=values["--dataflow-model-input-nodes"] {
            let nodes=try JSONDecoder().decode([String].self,from:Data(raw.utf8))
            let required=try JSONDecoder().decode([String].self,from:Data((values["--dataflow-required-nodes"] ?? "[]").utf8))
            guard nodes.count<=128,Set(nodes).count==nodes.count,Set(nodes).isSubset(of:Set(required)) else {throw WorkflowFileLeaseError.unsafePath}
        }
        if let raw=values["--dataflow-required-nodes"] {
            let nodes=try JSONDecoder().decode([String].self,from:Data(raw.utf8));guard nodes.count<=128,Set(nodes).count==nodes.count else {throw WorkflowFileLeaseError.unsafePath}
        }
        if let raw=values["--defer-database-writes"],raw != "true" {throw WorkflowFileLeaseError.unsafePath}
        if let raw=values["--dataflow-capture-outputs"],!["true","false"].contains(raw) {throw WorkflowFileLeaseError.unsafePath}
        if let raw=values["--dataflow-measurements"],!["true","false"].contains(raw) {throw WorkflowFileLeaseError.unsafePath}
        if let raw=values["--dataflow-live-previews"],!["true","false"].contains(raw) {throw WorkflowFileLeaseError.unsafePath}
        if let raw = values["--dataflow-cache-limit-bytes"] {
            guard phase == .preparation, ["materialize", "reuse"].contains(values["--dataflow-cache-policy"] ?? ""),
                let limit = Int(raw), String(limit) == raw, ExecutionPreferences.preparationCacheByteBounds.contains(limit) else { throw WorkflowFileLeaseError.unsafePath }
        }
        if let raw = values["--training-control-json"] { _ = try JSONDecoder().decode(TrainingControlPolicy.self,from:Data(raw.utf8)) }
        if let raw = values["--training-progress-json"] { _ = try TrainingReportingPolicy.decode(Data(raw.utf8)) }
        if let capture = values["--sample-capture-json"] {
            guard capture.utf8.count <= 1024 else { throw WorkflowFileLeaseError.unsafePath }
            _ = try JSONDecoder().decode(TrainingSampleCapturePolicy.self, from: Data(capture.utf8))
        }
        if let raw = values["--initialization-recipe-json"] {
            guard phase == .training, raw.utf8.count <= InitializationConfiguration.maximumRecipeBytes else { throw WorkflowFileLeaseError.unsafePath }
            let value = try JSONDecoder().decode(JSONValue.self, from: Data(raw.utf8))
            let configuration = try InitializationConfiguration.decodeRecipe(value)
            guard try WorkflowPhaseArtifacts.encode(value) == WorkflowPhaseArtifacts.encode(configuration.recipeValue) else { throw WorkflowFileLeaseError.unsafePath }
        }
        for key in ["--dataset-recipe-json","--loader-recipe-json"] {
            guard let raw=values[key] else {continue}
            guard phase == .training,raw.utf8.count<=32768,values["--prepared-dataflow"] != nil else {throw WorkflowFileLeaseError.unsafePath}
            let value=try JSONDecoder().decode(JSONValue.self,from:Data(raw.utf8))
            let canonical=try key == "--dataset-recipe-json" ? DatasetConfiguration.decodeRecipe(value).recipeValue:DataLoaderConfiguration.decodeRecipe(value).recipeValue
            guard try WorkflowPhaseArtifacts.encode(value) == WorkflowPhaseArtifacts.encode(canonical) else {throw WorkflowFileLeaseError.unsafePath}
            if key == "--loader-recipe-json" {
                let loader=try DataLoaderConfiguration.decodeRecipe(value)
                guard values["--batch-size"] == String(loader.batchSize) else {throw WorkflowFileLeaseError.unsafePath}
            }
        }
        if let raw = values["--gradient-policy-json"] {
            guard phase == .training, let permission = values["--derivative-capture-json"] else {
                throw WorkflowFileLeaseError.unsafePath
            }
            let value = try JSONDecoder().decode(JSONValue.self, from: Data(raw.utf8))
            let configuration = try GradientPolicyConfiguration.decodeRecipe(value)
            guard try WorkflowPhaseArtifacts.encode(value) == WorkflowPhaseArtifacts.encode(configuration.recipeValue) else {
                throw WorkflowFileLeaseError.unsafePath
            }
            let capture = try DerivativeCapturePolicy.decode(Data(permission.utf8))
            try capture.validate(gradientPolicy: configuration)
            try configuration.validateObjectiveCompatibility(reduction: objectiveReduction, scalarization: values["--objective-scalarization"] ?? "mean")
        } else if values["--derivative-capture-json"] != nil {
            throw WorkflowFileLeaseError.unsafePath
        }
        if let raw = values["--trainability-json"] {
            guard phase == .training else { throw WorkflowFileLeaseError.unsafePath }
            let value = try JSONDecoder().decode(JSONValue.self, from: Data(raw.utf8))
            let configuration = try TrainabilityConfiguration.decodeRecipe(value)
            guard try WorkflowPhaseArtifacts.encode(value) == WorkflowPhaseArtifacts.encode(configuration.recipeValue) else {
                throw WorkflowFileLeaseError.unsafePath
            }
        }
        if let raw=values["--optimizer-recipe-json"] {
            guard raw.utf8.count<=8192 else {throw WorkflowFileLeaseError.unsafePath}
            _ = try OptimizerConfiguration.decodeRecipe(JSONDecoder().decode(JSONValue.self,from:Data(raw.utf8)))
        }
        if let raw=values["--scheduler-recipe-json"] {
            guard raw.utf8.count<=8192,values["--optimizer-recipe-json"] != nil else {throw WorkflowFileLeaseError.unsafePath}
            _ = try SchedulerConfiguration.decodeRecipe(JSONDecoder().decode(JSONValue.self,from:Data(raw.utf8)))
        }
        guard values["--resume-checkpoint"] == nil || values["--finetune-checkpoint"] == nil else {throw WorkflowFileLeaseError.unsafePath}
        let phaseDirectory=workspace.appendingPathComponent("workflow-runs/\(projectID)/\(directoryID)/\(phase.rawValue)")
        guard values["--output"] == phaseDirectory.appendingPathComponent("artifacts").path else {throw WorkflowFileLeaseError.unsafePath}
        if phase == .preparation {
            guard values["--execute-dataflow"] == phaseDirectory.appendingPathComponent("request.json").path,values["--source-bindings"] == phaseDirectory.appendingPathComponent("bindings.json").path else {throw WorkflowFileLeaseError.unsafePath}
            if let path=values["--destination-bindings"] {guard path == phaseDirectory.appendingPathComponent("destinations.json").path else {throw WorkflowFileLeaseError.unsafePath}}
        } else {guard values["--graph"] == phaseDirectory.appendingPathComponent("request.json").path else {throw WorkflowFileLeaseError.unsafePath}}
    }
}
