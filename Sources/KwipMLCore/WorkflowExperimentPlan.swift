import Foundation

/// Explicit ordered values, owned by Train settings or a retained legacy Run setup.
/// Version one varies fresh-training controls; data/model searches need their own
/// validation and partition policy before they can be added to this contract.
public struct WorkflowExperimentAxis: Codable, Equatable {
    public let field: String
    public let values: [JSONValue]
    public init(field: String, values: [JSONValue]) { self.field=field; self.values=values }
}
public struct WorkflowExperimentBudget: Codable, Equatable {
    public let maximumRuns: Int
    public let maximumTotalEpochs: Int
    public init(maximumRuns: Int, maximumTotalEpochs: Int) {
        self.maximumRuns=maximumRuns; self.maximumTotalEpochs=maximumTotalEpochs
    }
}
public struct WorkflowExperimentVariant: Equatable {
    public let ordinal: Int
    public let overrides: [String: JSONValue]
    public let document: WorkflowDocument
    public let execution: WorkflowExecutionPlan
}
/// Pure preview planning. Performs no I/O or execution, changes no live document,
/// and makes no claim that private sources are frozen or devices are available.
public struct WorkflowExperimentPlan: Equatable {
    public let source: WorkflowDocument
    public let setup: WorkflowNodeAddress
    public let axes: [WorkflowExperimentAxis]
    public let budget: WorkflowExperimentBudget
    public let variants: [WorkflowExperimentVariant]
    public let totalEpochs: Int

    public static func build(_ source: WorkflowDocument, train: WorkflowNodeAddress,
                             axes: [WorkflowExperimentAxis], budget: WorkflowExperimentBudget,
                             modelRegistry: ComponentRegistry, dataRegistry: DataFlowOperatorRegistry) throws -> Self {
        func reject(_ message: String) -> DataFlowContractError { .init(message) }
        try WorkflowExperimentSpecification(axes:axes,budget:budget).validate()
        let baseline=try WorkflowCompiler.compile(source,train:train,modelRegistry:modelRegistry,dataRegistry:dataRegistry,contractVersion:source.schemaVersion>=17 ? 15:source.schemaVersion>=16 ? 14:source.schemaVersion>=15 ? 13:source.schemaVersion>=13 ? 12 : (source.schemaVersion>=12 ? 11 : (source.schemaVersion>=11 ? 10 : (source.schemaVersion>=10 ? 9 : (source.schemaVersion>=9 ? 8 : (source.schemaVersion>=8 ? 7 : (source.schemaVersion>=7 ? 6 : (source.schemaVersion>=6 ? 5 : (source.schemaVersion >= 4 ? 4:3)))))))))
        guard baseline.trainingMode == .fresh else {
            throw reject("Experiment grids currently require fresh training. Resume and fine-tuning need an explicit parent policy.")
        }
        let authority=try WorkflowTrainRunSetup.resolve(in:source,train:train)
        guard let setupIndex=source.nodes.firstIndex(where:{$0.address==authority.address}) else {throw reject("The selected Train run settings are unavailable.")}
        let original=authority.configuration
        if baseline.optimizerRecipe != nil,axes.contains(where:{$0.field=="learningRate"}) {throw reject("This Train uses an optimizer recipe. Compare optimizer.lr instead of the inactive Train learning rate.")}
        if baseline.optimizerRecipe == nil,axes.contains(where:{$0.field=="optimizer.lr"}) {throw reject("Choose an optimizer before comparing optimizer.lr; use learningRate for the legacy Adam recipe.")}
        let loaderBatchAxis=baseline.loaderRecipe != nil && axes.contains(where:{$0.field=="batchSize"})
        if loaderBatchAxis {
            guard baseline.loaderRecipe?.source==train,baseline.inlineDataLoadingRoles.contains("loader") else {
                throw reject("Batch size belongs to the connected loader. Upgrade it into Train before comparing batch sizes.")
            }
        }
        var combinations:[[String:JSONValue]]=[[:]]
        for axis in axes {
            combinations=combinations.flatMap { existing in
                axis.values.map { value in var next=existing; next[axis.field]=value; return next }
            }
        }
        // Determine total epoch work before retaining expanded documents.
        var total=0
        for combination in combinations {
            let config=original.merging(combination.filter {$0.key != "optimizer.lr" && !(loaderBatchAxis && $0.key=="batchSize")}) { _, new in new }
            let settings=try WorkflowControlContracts.runSettings(config,address:authority.address)
            guard settings.epochs <= budget.maximumTotalEpochs-total else {
                throw reject("Experiment work exceeds the chosen total epoch budget.")
            }
            total += settings.epochs
        }
        let variants=try combinations.enumerated().map { index, overrides in
            var document=source
            let updated=original.merging(overrides.filter {$0.key != "optimizer.lr" && !(loaderBatchAxis && $0.key=="batchSize")}) { _,new in new }
            if source.schemaVersion>=16 {
                guard case .train(let config)=document.nodes[setupIndex].payload else {throw reject("The inline Train settings authority changed.")}
                document.nodes[setupIndex].payload = .train(config:try WorkflowTrainRunSetup.setting(updated,in:config))
            } else {document.nodes[setupIndex].payload = .runSetup(config:updated)}
            if loaderBatchAxis,let batchSize=overrides["batchSize"] {
                guard let position=document.nodes.firstIndex(where:{$0.address==train}),case .train(var config)=document.nodes[position].payload,
                      case .object(var batching)=config["batching"] else {throw reject("The inline batching authority changed.")}
                batching["batch_size"]=batchSize
                _ = try DataLoaderConfiguration(values:batching)
                config["batching"] = .object(batching)
                document.nodes[position].payload = .train(config:config)
            }
            if let rate=overrides["optimizer.lr"] {
                guard let recipe=baseline.optimizerRecipe,let position=document.nodes.firstIndex(where:{$0.address==recipe.source}) else {throw reject("The selected optimizer source is unavailable.")}
                if baseline.inlineTrainingRecipeRoles.contains("optimizer"),recipe.source == train,case .train(var config)=document.nodes[position].payload {
                    var values=recipe.configuration.values;values["lr"]=rate
                    config["optimizer"] = try OptimizerConfiguration(componentID:recipe.configuration.componentID,values:values).recipeValue
                    document.nodes[position].payload = .train(config:config)
                } else if case .optimizerRecipe(let type,let version,var config)=document.nodes[position].payload {
                    config["lr"]=rate;document.nodes[position].payload = .optimizerRecipe(type:type,configurationVersion:version,config:config)
                } else {throw reject("The optimizer origin does not match its captured inline or legacy authority.")}
            }
            let execution=try WorkflowCompiler.compile(document,train:train,modelRegistry:modelRegistry,dataRegistry:dataRegistry,contractVersion:source.schemaVersion>=17 ? 15:source.schemaVersion>=16 ? 14:source.schemaVersion>=15 ? 13:source.schemaVersion>=13 ? 12 : (source.schemaVersion>=12 ? 11 : (source.schemaVersion>=11 ? 10 : (source.schemaVersion>=10 ? 9 : (source.schemaVersion>=9 ? 8 : (source.schemaVersion>=8 ? 7 : (source.schemaVersion>=7 ? 6 : (source.schemaVersion>=6 ? 5 : (source.schemaVersion >= 4 ? 4:3)))))))))
            return WorkflowExperimentVariant(ordinal:index+1,overrides:overrides,document:document,execution:execution)
        }
        return .init(source:source,setup:authority.address,axes:axes,budget:budget,variants:variants,totalEpochs:total)
    }
}

public struct WorkflowExperimentSpecification: Codable, Equatable {
    public let schemaVersion: Int
    public let axes: [WorkflowExperimentAxis]
    public let budget: WorkflowExperimentBudget
    public init(axes:[WorkflowExperimentAxis],budget:WorkflowExperimentBudget) {
        schemaVersion=1;self.axes=axes;self.budget=budget
    }
    public func validate() throws {
        func reject(_ message:String)->DataFlowContractError {.init(message)}
        guard schemaVersion == 1 else {throw reject("Unsupported experiment specification version.")}
        guard (1...32).contains(budget.maximumRuns), (1...320_000).contains(budget.maximumTotalEpochs) else {
            throw reject("Experiment limits require 1–32 runs and 1–320000 total epochs.")
        }
        let fields=WorkflowControlContracts.experimentFields
        guard !axes.isEmpty, axes.count <= fields.count, Set(axes.map(\.field)).count == axes.count else {
            throw reject("Choose distinct experiment fields with at least one dimension.")
        }
        var count=1
        for axis in axes {
            guard let field=fields.first(where: {$0.key == axis.field}), !axis.values.isEmpty,
                  axis.values.count <= budget.maximumRuns, axis.values.allSatisfy(field.accepts) else {
                throw reject("Experiment field \(axis.field) has unsupported or invalid values.")
            }
            let unique: Bool
            if field.key=="optimizer.lr" || field.key=="learningRate" {
                unique=Set(axis.values.compactMap(\.doubleValue)).count == axis.values.count
            } else { unique=Set(axis.values).count == axis.values.count }
            guard unique else { throw reject("Experiment field \(axis.field) repeats a value.") }
            guard count <= budget.maximumRuns / axis.values.count else {
                throw reject("Experiment combinations exceed the chosen run limit. Reduce values or increase the limit.")
            }
            count *= axis.values.count
        }
    }
    public static func decode(_ value:JSONValue)throws->Self {
        func object(_ value:JSONValue,_ keys:Set<String>)throws->[String:JSONValue] {
            guard case .object(let fields)=value,Set(fields.keys)==keys else {throw DataFlowContractError("Unknown or missing experiment specification field.")}
            return fields
        }
        let root=try object(value,["schemaVersion","axes","budget"])
        guard root["schemaVersion"] == .integer(1),case .array(let axes)=root["axes"],axes.count<=6 else {throw DataFlowContractError("Invalid experiment specification version or axes.")}
        for axis in axes {_ = try object(axis,["field","values"])}
        _ = try object(root["budget"]!,["maximumRuns","maximumTotalEpochs"])
        let result=try JSONDecoder().decode(Self.self,from:JSONEncoder().encode(value))
        try result.validate();return result
    }
}
