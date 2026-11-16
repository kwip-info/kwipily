import Foundation
import Combine
import KwipMLCore

@MainActor final class WorkflowControlNodeContext:ObservableObject {
    @Published var selectedModelInputName="input"
    let authority:WorkflowNodeEditingAuthority
    private var observation:AnyCancellable?
    init(owner:WorkflowAuthoringStore,address:WorkflowNodeAddress)throws {
        guard address.domain == .control || owner.document.nodes.first(where:{$0.address==address}).map(WorkflowModelBoundary.isDefinition)==true else {throw DataFlowContractError("Select a workflow control node.")}
        authority=try .init(owner:owner,address:address)
        selectedModelInputName=owner.document.namedModelInputs?[address.localID]?.keys.sorted().first ?? "input"
        observation=authority.objectWillChange.sink {[weak self] _ in self?.objectWillChange.send()}
    }
    var contextIdentity:String {authority.contextIdentity}
    var isCurrent:Bool {authority.isCurrent}
    var isBusy:Bool {authority.isBusy}
    var configuration:[String:JSONValue] {authority.configuration}
    var authoredName:String {authority.node?.name ?? ""}
    var statusMessage:String {authority.message ?? ""}
    var isGameSession:Bool {if case .gameSession=authority.node?.payload {return true};return false}
    var isGenerateText:Bool {if case .generateText=authority.node?.payload {return true};return false}
    var isSaveModel:Bool {if case .saveModel=authority.node?.payload {return true};return false}
    var saveModelConfiguration:WorkflowSaveModelConfiguration? {
        guard isSaveModel,isCurrent,authority.owner.document.schemaVersion>=17 else {return nil}
        return try? WorkflowSaveModelConfiguration.decode(configuration)
    }
    @discardableResult func setSaveModelFormat(_ format:WorkflowSaveModelConfiguration.Format,expected:[String:JSONValue])->Bool {
        guard isSaveModel,authority.owner.document.schemaVersion>=17,authority.accepts(expected) else {return false}
        do {
            var next=expected;next["format"] = .string(format.rawValue)
            if format != .huggingFace {next.removeValue(forKey:"shardBytes");next.removeValue(forKey:"content")}
            _=try WorkflowSaveModelConfiguration.decode(next)
            return authority.configure(next,expected:expected)
        } catch {authority.message=error.localizedDescription;return false}
    }
    var isFrozenModel:Bool {if case .frozenModel=authority.node?.payload {return true};return false}
    var isEvaluateTest:Bool {if case .evaluateTest=authority.node?.payload {return true};return false}
    var evaluationSettings:WorkflowEvaluationSettings? {
        guard isEvaluateTest,isCurrent,authority.owner.document.schemaVersion>=5 else {return nil}
        return try? WorkflowEvaluationSettings.decode(configuration)
    }
    @discardableResult func setEvaluationValue(_ key:String,value:JSONValue,expected:[String:JSONValue])->Bool {
        guard isEvaluateTest,authority.owner.document.schemaVersion>=5,authority.accepts(expected) else {return false}
        var next=expected;next[key]=value
        do {
            _=try WorkflowEvaluationSettings.decode(next)
            return authority.configure(next,expected:expected)
        } catch {authority.message=error.localizedDescription;return false}
    }
    @discardableResult func setEvaluationPartition(_ role:String,expected:[String:JSONValue])->Bool {
        guard isEvaluateTest,authority.accepts(expected),["validation","test"].contains(role) else {return false}
        var next=expected
        if role == "validation" {next["settingsVersion"]=2;next["partitionRole"] = .string(role)}
        else {next.removeValue(forKey:"settingsVersion");next.removeValue(forKey:"partitionRole")}
        do {_=try WorkflowEvaluationSettings.decode(next);return authority.configure(next,expected:expected)}
        catch {authority.message=error.localizedDescription;return false}
    }
    var isEstimator:Bool {if case .estimator=authority.node?.payload {return true};return false}
    var learningRateIsInactive:Bool {
        if isTrain {return configuration["optimizer"] != nil || authority.owner.document.edges.contains {$0.to.address == authority.address && $0.to.port == "optimizer" && $0.kind == .optimizerRecipe}}
        guard isRunSetup else {return false}
        let trains=authority.owner.document.edges.filter {$0.from.address==authority.address && $0.to.port=="configuration"}.map(\.to.address)
        return !trains.isEmpty && trains.allSatisfy {train in
            authority.owner.document.nodes.contains {node in guard node.address == train,case .train(let config)=node.payload else {return false};return config["optimizer"] != nil}
            || authority.owner.document.edges.contains {$0.to.address==train && $0.to.port=="optimizer" && $0.kind == .optimizerRecipe}
        }
    }
    var isRunSetup:Bool {if case .runSetup=authority.node?.payload {return true};return false}
    /// Raw saved settings; displaying defaults never writes them into the document.
    var runConfiguration:[String:JSONValue]? {
        guard isCurrent else {return nil}
        if isRunSetup {return configuration}
        guard isTrain else {return nil}
        return try? WorkflowTrainRunSetup.resolve(in:authority.owner.document,train:authority.address).configuration
    }
    func mappingRequirements()throws->[WorkflowMappingRequirement] {
        guard isMapping,isCurrent else {return []}
        return try WorkflowMappingRequirements.resolve(authority.owner.document,mapping:authority.address,
            modelRegistry:authority.owner.modelRegistry,dataRegistry:authority.owner.dataRegistry)
    }
    var requiresBagMapping:Bool {
        guard isMapping,isCurrent else {return false}
        if mappingConfiguration["bagInput"] != nil {return true}
        if (try? mappingRequirements().contains {$0.bagRequirement != nil}) == true {return true}
        // Keep a broken, connected bag model on its repair route, rather than silently
        // switching to a dense editor when validation cannot yet produce a contract.
        let document=authority.owner.document
        let trains=isTrain ? Set([authority.address]):Set(document.edges.filter {$0.from.address==authority.address && $0.to.port=="data"}.map(\.to.address))
        var pending=document.edges.filter {trains.contains($0.to.address) && $0.to.port=="model"}.map(\.from.address)
        var visited=Set<WorkflowNodeAddress>()
        while let address=pending.popLast() {
            guard visited.insert(address).inserted else {continue}
            if let node=document.nodes.first(where:{$0.address==address}),case .modelPrimitive("bag_input",_,_)=node.payload {return true}
            pending += document.edges.filter {$0.to.address==address && $0.from.address.domain == .model}.map(\.from.address)
        }
        return false
    }
    @discardableResult func applyBagMapping(_ mapping:BagInputMapping,target:String,expected:[String:JSONValue])->Bool {
        guard isMapping,!hasObjectiveMapping,authority.accepts(expected) else {return false}
        do {
            for requirement in try bagMappingRequirements() {
                guard let bag=requirement.bagRequirement else {return false}
                try mapping.validate(sourceConfiguration:bag.configuration,targetColumn:target)
            }
            let next:[String:JSONValue]=["bagInput":mapping.value,"targetColumn":.string(target)]
            _=try WorkflowControlContracts.bagInputMapping(next,address:authority.address)
            return configureMapping(next,expected:expected)
        } catch {authority.message=error.localizedDescription;return false}
    }
    var isMapping:Bool {if case .trainingData=authority.node?.payload {return true};return isTrain && authority.owner.document.schemaVersion>=18}
    var mappingConfiguration:[String:JSONValue] {
        if isTrain,case .object(let value)=configuration["dataMapping"] {return value}
        return isTrain ? [:]:configuration
    }
    @discardableResult func configureMapping(_ value:[String:JSONValue],expected:[String:JSONValue])->Bool {
        do {
            try WorkflowTrainMapping.validate(value)
            if isTrain {var next=expected;next["dataMapping"] = .object(value);return authority.configure(next,expected:expected)}
            return authority.configure(value,expected:expected)
        } catch {authority.message=error.localizedDescription;return false}
    }
    @discardableResult func configureSequenceMapping(_ value:[String:JSONValue],objective:LossConfiguration,expected:[String:JSONValue])->Bool {
        do {
            guard isTrain else {throw DataFlowContractError("Next-token mapping belongs to Train.")}
            try WorkflowTrainMapping.validate(value)
            guard case .object(let raw)=value["lossTargetMapping"] else {throw DataFlowContractError("Next-token targets are missing.")}
            let target=try JSONDecoder().decode(LossTargetMapping.self,from:JSONEncoder().encode(JSONValue.object(raw)))
            try objective.validate(targetMapping:target)
            var next=expected;next["objective"]=objective.recipeValue;next["dataMapping"] = .object(value)
            return authority.configure(next,expected:expected)
        } catch {authority.message=error.localizedDescription;return false}
    }
    /// Nil means invalid/unavailable, never silently interpreted as Off.
    var preparationCachePolicy:WorkflowPreparationCachePolicy? {
        guard let run=runConfiguration else {return nil}
        return try? WorkflowControlContracts.preparationCachePolicy(run,schemaVersion:authority.owner.document.schemaVersion)
    }
    @discardableResult func setPreparationCachePolicy(_ policy:WorkflowPreparationCachePolicy,expected:[String:JSONValue])->Bool {
        guard (isRunSetup || isTrain),authority.accepts(expected) else {return false}
        return authority.applyTransaction {lease in
            authority.owner.setPreparationCachePolicy(policy,for:authority.address,lease:lease)
        }
    }
    private var experimentSetupAddress:WorkflowNodeAddress? {
        guard isCurrent else {return nil}
        if isRunSetup {return authority.address}
        guard isTrain else {return nil}
        if authority.owner.document.schemaVersion >= 16 {return authority.address}
        let links=authority.owner.document.edges.filter {$0.to.address == authority.address && $0.to.port == "configuration"}
        guard links.count == 1,let link=links.first,link.kind == .runConfiguration,link.from.port == "configuration",
              authority.owner.document.nodes.contains(where: {guard $0.address == link.from.address else {return false};if case .runSetup = $0.payload {return true};return false}) else {return nil}
        return link.from.address
    }
    var experimentConfiguration:[String:JSONValue]? {
        guard let setup=experimentSetupAddress else {return nil}
        if setup == authority.address,isTrain {return configuration}
        guard let node=authority.owner.document.nodes.first(where:{$0.address == setup}),case .runSetup(let config)=node.payload else {return nil}
        return config
    }
    var experimentSpecification:WorkflowExperimentSpecification? {
        guard let value=runConfiguration?["experiment"] else {return nil}
        return try? WorkflowExperimentSpecification.decode(value)
    }
    @discardableResult func setExperimentSpecification(_ specification:WorkflowExperimentSpecification?,expected:[String:JSONValue])->Bool {
        guard !isBusy,let setup=experimentSetupAddress,experimentConfiguration == expected else {return false}
        return authority.applyTransaction {lease in
            authority.owner.setExperimentSpecification(specification,for:setup,lease:lease)
        }
    }
    var experimentTrainNodes:[WorkflowNode] {
        guard let setup=experimentSetupAddress else {return []}
        if isTrain {return authority.node.map {[$0]} ?? []}
        let targets=Set(authority.owner.document.edges.filter {$0.from.address == setup && $0.from.port == "configuration" && $0.to.port == "configuration" && $0.kind == .runConfiguration}.map { $0.to.address })
        return authority.owner.document.nodes.filter {node in guard targets.contains(node.address) else {return false};if case .train = node.payload {return true};return false}
    }
    func previewExperiment(_ specification:WorkflowExperimentSpecification,train:WorkflowNodeAddress,
                           expected:[String:JSONValue])throws->WorkflowExperimentPlan {
        guard !isBusy,experimentConfiguration == expected,experimentTrainNodes.contains(where:{$0.address == train}) else {
            throw DataFlowContractError("Choose the current Train before previewing experiments.")
        }
        return try WorkflowExperimentPlan.build(authority.owner.document,train:train,
            axes:specification.axes,budget:specification.budget,
            modelRegistry:authority.owner.modelRegistry,dataRegistry:authority.owner.dataRegistry)
    }
    var isTrain:Bool {if case .train=authority.node?.payload {return true};return false}
    @discardableResult func applyTrainDataAccess(_ next:[String:JSONValue],expected:[String:JSONValue])->Bool {
        guard isTrain,authority.accepts(expected) else {return false}
        let keys:Set<String>=["datasetAccess","batching"]
        guard next.filter({!keys.contains($0.key)}) == expected.filter({!keys.contains($0.key)}) else {authority.message="Preserve unrelated Train settings when editing data access.";return false}
        do {
            _=try WorkflowTrainDataLoading(configuration:next,schemaVersion:max(13,authority.owner.document.schemaVersion))
            guard !authority.owner.document.edges.contains(where: {$0.to.address == authority.address && ["dataset","loader"].contains($0.to.port)}) else {throw DataFlowContractError("Reopen this project to migrate its saved data-access settings into Train.")}
            return authority.configure(next,expected:expected)
        } catch {authority.message=error.localizedDescription;return false}
    }
    /// An unavailable context never silently falls back to fresh training.
    var trainingMode:WorkflowTrainingMode? {
        guard isTrain,isCurrent else {return nil}
        return try? WorkflowControlContracts.trainingMode(configuration,schemaVersion:authority.owner.document.schemaVersion)
    }
    @discardableResult func setTrainingMode(_ mode:WorkflowTrainingMode,expected:[String:JSONValue])->Bool {
        guard isTrain,authority.accepts(expected) else {return false}
        return authority.applyTransaction {lease in
            authority.owner.setTrainingMode(mode,for:authority.address,lease:lease)
        }
    }
    var gradientConfiguration:GradientCaptureConfiguration? {
        guard let value=runConfiguration?["gradientCapture"],value != .null,
              let data=try? JSONEncoder().encode(value) else {return nil}
        return try? JSONDecoder().decode(GradientCaptureConfiguration.self,from:data)
    }
    @discardableResult func setRunValue(_ key:String,value:JSONValue,expected:[String:JSONValue])->Bool {
        guard (isRunSetup || isTrain),authority.accepts(expected),WorkflowControlContracts.runFields.contains(where:{$0.key == key}) else {return false}
        guard var next=runConfiguration else {return false};next[key]=value
        return applyRun(next,expected:expected)
    }
    var trainingControlPolicy:TrainingControlPolicy? {
        guard let run=runConfiguration else {return nil}
        return try? WorkflowControlContracts.runSettings(run,address:authority.address).trainingControl ?? .factory
    }
    @discardableResult func setTrainingControl(_ policy:TrainingControlPolicy,expected:[String:JSONValue])->Bool {
        guard isTrain,authority.accepts(expected),var next=runConfiguration else {return false}
        do {try policy.validate();next["trainingControl"]=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(policy));return applyRun(next,expected:expected)}
        catch {authority.message=error.localizedDescription;return false}
    }
    private func applyRun(_ next:[String:JSONValue],expected:[String:JSONValue])->Bool {
        do {
            _=try WorkflowControlContracts.runSettings(next,address:authority.address)
            if isTrain {
                guard authority.owner.document.schemaVersion >= 16 else {throw DataFlowContractError("Reopen this project to move Run setup into Train.")}
                let updated=try WorkflowTrainRunSetup.setting(next,in:expected,schemaVersion:authority.owner.document.schemaVersion)
                return authority.configure(updated,expected:expected)
            }
            return authority.configure(next,expected:expected)
        }
        catch {authority.message=error.localizedDescription;return false}
    }
    @discardableResult func setGradientCapture(_ value:GradientCaptureConfiguration?,expected:[String:JSONValue])->Bool {
        guard (isRunSetup || isTrain),authority.accepts(expected) else {return false}
        do {
            try value?.validate()
            guard var next=runConfiguration else {return false}
            next["gradientCapture"]=try value.map {try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode($0))} ?? .null
            return applyRun(next,expected:expected)
        } catch {authority.message=error.localizedDescription;return false}
    }
    @discardableResult func applyObjectiveMapping(_ mapping:LossTargetMapping,bagMapping:BagInputMapping?=nil,expected:[String:JSONValue])->Bool {
        guard isMapping,hasObjectiveMapping,authority.accepts(expected) else {return false}
        do {
            // Resolve the current actual consumer/recipe/model again; a retained editor never
            // certifies a different objective or silently falls back to legacy mapping.
            let current=try LossTargetMappingInspectorContext(context:self)
            try current.validate(mapping,bagMapping:bagMapping)
            let value=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(mapping))
            var next:[String:JSONValue] = ["lossTargetMapping":value]
            if requiresBagMapping {
                guard let bagMapping else {throw DataFlowContractError("Preserve grouped source bindings when applying objective roles.")}
                next["bagInput"]=bagMapping.value
                _=try WorkflowControlContracts.bagInputMapping(next,address:authority.address)
            } else {
                guard bagMapping == nil else {throw DataFlowContractError("Connect a grouped input model before mapping bag columns.")}
                _=try WorkflowControlContracts.lossTargetMapping(next,address:authority.address)
            }
            return configureMapping(next,expected:expected)
        } catch {authority.message=error.localizedDescription;return false}
    }
    @discardableResult func applyMapping(features:[String],target:String,expected:[String:JSONValue])->Bool {
        guard isMapping,authority.accepts(expected) else {return false}
        guard !requiresBagMapping else {authority.message="Use grouped source mapping to preserve category IDs, lengths and item weights.";return false}
        let next:[String:JSONValue] = ["featureColumns":.array(features.map(JSONValue.string)),"targetColumn":.string(target)]
        do {_=try WorkflowControlContracts.trainingMapping(next,address:authority.address);return configureMapping(next,expected:expected)}
        catch {authority.message=error.localizedDescription;return false}
    }
    @discardableResult func rename(_ name:String)->Bool {authority.rename(name)}
    func numericOrigin(key:String)->String {authority.numericOrigin(key:key)}
    func currentNumericOrigin(key:String)->String? {authority.currentNumericOrigin(key:key)}
    func setNumericPending(origin:String,key:String,pending:Bool) {authority.setNumericPending(origin:origin,key:key,pending:pending)}
    func retire() {authority.retire()}
}
