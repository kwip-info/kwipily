import Foundation

/// Discovery for the existing Train authorities, not a second execution validator.
/// Examples are serialized native values; model/column compatibility still needs
/// WorkflowCompiler and captured data. Empty authoring defaults are not runnable.
public enum WorkflowTrainConfigurationDiscovery {
    private static func encoded<T:Encodable>(_ value:T)throws->JSONValue {try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(value))}
    /// Structural discovery from serialized native contracts. Semantic validation
    /// remains in their native initializers/compiler, never in this projection.
    private static func shape(_ value:JSONValue)->JSONValue {
        switch value {
        case .object(let fields):return .object(["type":.string("object"),"properties":.object(fields.mapValues(shape)),"required":.array(fields.keys.sorted().map(JSONValue.string)),"additionalProperties":false])
        case .array(let items):return .object(["type":.string("array"),"items":items.first.map(shape) ?? .object([:])])
        case .string:return .object(["type":.string("string")])
        case .integer:return .object(["type":.string("integer")])
        case .number:return .object(["type":.string("number")])
        case .bool:return .object(["type":.string("boolean")])
        case .null:return .object(["type":.string("null")])
        }
    }
    private static func recipeSchema(_ recipe:JSONValue)->JSONValue {
        guard case .object(let values)=recipe,case .object(var schema)=shape(recipe),case .object(var properties)=schema["properties"] else{return .object([:])}
        properties["componentID"] = .object(["const":values["componentID"]!])
        if case .object(var config)=properties["config"] {config.removeValue(forKey:"required");properties["config"] = .object(config)}
        schema["properties"] = .object(properties);return .object(schema)
    }
    public static func supervisedExample()throws->[String:JSONValue] {
        let mapping=LossTargetMapping(componentID:"loss_mse",inputColumns:["feature_1","feature_2"],roles:[.init(role:"target",columns:["target_1","target_2"],sampleShape:[2],dtype:.float32)])
        try mapping.validate()
        return ["dataMapping":.object(["lossTargetMapping":try encoded(mapping)]),
                "runSetup":.object(["device":.string("cpu"),"epochs":3,"batchSize":16,"seed":42,"learningRate":0.001]),
                "objective":try LossConfiguration(componentID:"loss_mse",values:[:]).recipeValue,
                "optimizer":try OptimizerConfiguration(componentID:"optimizer_adam",values:[:]).recipeValue]
    }
    public static func contract()throws->JSONValue {
        var fields:[String:JSONValue]=[:]
        for field in WorkflowControlContracts.runFields {
            var item:[String:JSONValue]=["title":.string(field.title),"default":field.defaultValue]
            switch field.kind {
            case .integer(let bounds):item.merge(["type":.string("integer"),"minimum":.integer(bounds.lowerBound),"maximum":.integer(bounds.upperBound)]) {_,new in new}
            case .positiveDecimal(let max):item.merge(["type":.string("number"),"exclusiveMinimum":0,"maximum":.number(max)]) {_,new in new}
            case .nonnegativeDecimal(let max):item.merge(["type":.string("number"),"minimum":0,"maximum":.number(max)]) {_,new in new}
            case .choice(let choices):item.merge(["type":.string("string"),"enum":.array(choices.map(JSONValue.string))]) {_,new in new}
            }
            fields[field.key] = .object(item)
        }
        fields["trainingControl"]=TrainingControlPolicy.configurationSchema
        fields["preparationCachePolicy"] = .object(["type":.string("string"),"enum":.array(WorkflowPreparationCachePolicy.allCases.map{.string($0.rawValue)})])
        let example=try supervisedExample()
        let named=NamedInputPlan(inputs:[.init(name:"observations",nodeID:"model_input",sampleShape:[2],columns:["feature_1","feature_2"])],roles:[.init(name:"target",sampleShape:[2],columns:["target_1","target_2"])])
        try named.validate()
        return .object([
            "configurationSchema":.object(["type":.string("object"),"required":.array(["runSetup","dataMapping"].map(JSONValue.string)),"properties":.object([
                "runSetup":.object(["type":.string("object"),"properties":.object(fields)]),
                "dataMapping":.object(["anyOf":.array([shape(.object(["featureColumns":.array([.string("feature_1")]),"targetColumn":.string("target")])),shape(example["dataMapping"]!)])]),
                "objective":.object(["oneOf":.array(try LossConfiguration.componentIDs.sorted().map {recipeSchema(try LossConfiguration(componentID:$0,values:[:]).recipeValue)})]),"optimizer":.object(["oneOf":.array(try OptimizerConfiguration.componentIDs.sorted().map {recipeSchema(try OptimizerConfiguration(componentID:$0,values:[:]).recipeValue)})])]),
                "description":.string("Supervised numeric mapping projection; additional native Train fields and named/sequence/bag mapping variants are described separately. Native validation is authoritative, including objective-specific config fields.")]),
            "configurationPaths":.object(["runSetup":.string("Train-owned run controls"),"dataMapping":.string("Train-owned ordered input and target mapping"),"objective":.string("LossConfiguration recipe envelope"),"optimizer":.string("OptimizerConfiguration recipe envelope")]),
            "runSetup":.object(["type":.string("object"),"properties":.object(fields),"additionalNativeControls":.array(["gradientCapture","experiment"].map(JSONValue.string)),"validationAuthority":.string("WorkflowTrainRunSetup.configuration / WorkflowControlContracts.runSettings")]),
            "dataMapping":.object([
                "simpleSingleTarget":.object(["featureColumns":.array(["feature_1","feature_2"].map(JSONValue.string)),"targetColumn":.string("target")]),
                "explicitObjective":example["dataMapping"]!,
                "namedInputs":.object(["namedInputPlan":try encoded(named),"lossTargetMapping":{if case .object(let m)=example["dataMapping"]! {return m["lossTargetMapping"]!};return .null}()]),
                "contract":.string("Use simple mapping for one target column, or lossTargetMapping for explicit typed/vector targets. Named inputs augment the explicit objective mapping and must match every declared Model Input name, shape and dtype; nodeID refers to its real model input. Sample shape excludes batch. Ordered numeric columns fill row-major. Never invent one targetColumn for a multi-column vector. Bag mapping and sequence objectives use separate native contracts."),
                "validationAuthorities":.array(["WorkflowTrainMapping","LossTargetMapping","NamedInputPlan","WorkflowMappingRequirements"].map(JSONValue.string))]),
            "objectiveRecipes":.array(try LossConfiguration.componentIDs.sorted().map {try LossConfiguration(componentID:$0,values:[:]).recipeValue}),
            "optimizerRecipes":.array(try OptimizerConfiguration.componentIDs.sorted().map {try OptimizerConfiguration(componentID:$0,values:[:]).recipeValue}),
            "supervisedExample":.object(example),
            "connections":.object(["training":.string("Connect training table to Train.training"),"validation":.string("Connect held-out table to Train.validation; same mapped schema"),"model":.string("Canvas modelDefinition output to Train.model. Mapping does not create a partition or model.")]),
            "scope":.string("Supervised canvas Train discovery. Recipes are native resolved defaults, not permission to use every objective with every model. The example needs real matching table columns and a model with two inputs and two float32 outputs. Preview checks authoring; workflow_validate and run qualify model/data compatibility. Empty default mapping is unfinished."),
            "separateModes":.object(["fineTuning":.string("Use fineTuning schema and verified loaded-model capabilities; supervised mapping does not describe prompt/response loss or exact resume."),"selfPlay":.string("Use selfPlay schema and game policy/value binding; not supervised table training.")]),
            "additionalNativeAuthorities":.array(["WorkflowTrainDataLoading","WorkflowTrainRecipes","WorkflowExperimentSpecification","GradientCaptureConfiguration","BagInputMapping"].map(JSONValue.string))])
    }
}
