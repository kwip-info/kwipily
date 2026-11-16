import Foundation

public enum DataLoadingLearningTask:String,CaseIterable,Codable {
    case indexed="dataset-indexed-observations",tensor="dataset-aligned-tensors",subset="dataset-selected-view",iterable="dataset-finite-stream"
    case batches="loader-complete-and-short-groups",shuffle="loader-repeatable-mixing",collation="loader-aligned-sequences",continuation="loader-continue-grouping"
    public var title:String {switch self {
    case .indexed:return "Revisit numbered observations";case .tensor:return "Keep inputs and targets together";case .subset:return "Choose an ordered view";case .iterable:return "Read a finite stream"
    case .batches:return "Make complete and short groups";case .shuffle:return "Mix rows reproducibly";case .collation:return "Keep sequence lengths aligned";case .continuation:return "Continue the same grouping"
    }}
    public var controlInstructions:String {switch self {
    case .indexed:return "Choose Indexed rows. Keep the index list empty."
    case .tensor:return "Choose Tensor tuple. Keep the index list empty. A tuple keeps the input and desired outcome together."
    case .subset:return "Choose Selected indices and enter 4, 1, 0 in that order. Positions start at zero and always refer to the original training partition."
    case .iterable:return "Choose Finite stream and set Maximum rows to 5. The same finite snapshot restarts for each pass."
    case .batches:return "Set Batch size to 4 and keep Drop final short group off. Use Fixed stack. Compare the observed groups with the example that drops the short group."
    case .shuffle:return "Set Batch size to 3, Mix rows on and Mixing seed to 42. Keep Drop final short group off."
    case .collation:return "Set Batch size to 4 and Combine rows to Padded sequence. Keep row mixing off; the supplied zero padding and true lengths remain together."
    case .continuation:return "Set Batch size to 4, Mix rows on, Mixing seed to 42, Reader processes to 2, Prefetch to 1, Timeout to 5 seconds and Combine rows to Whole bags. The checker resumes a real saved run on CPU."
    }}
    public var instructions:String {
        let meaning:String
        switch self {
        case .indexed:meaning="An observation is one input row with its desired outcome. Indexed access lets the reader revisit a numbered observation without changing its identity."
        case .tensor:meaning="A tensor is an array of numbers. A tensor tuple keeps each input row beside its desired outcome, even when rows are selected later."
        case .subset:meaning="A selected view reuses original rows in the requested order. Position zero means the first row of the training partition."
        case .iterable:meaning="A finite stream reads a bounded number of observations in order. It restarts from the same snapshot for each pass."
        case .batches:meaning="A batch is a group processed together. When the row count is not divisible by the batch size, the final group is shorter."
        case .shuffle:meaning="Mixing changes row order, while a seed makes that order repeatable. Inputs, desired outcomes and row IDs must move together."
        case .collation:meaning="Collation combines rows into a batch. Padding fills shorter sequences with zeros; true lengths tell the model which positions belong to the observation."
        case .continuation:meaning="A checkpoint saves training progress. Continuing it must preserve the next row order and learned settings. Reader processes load rows separately without changing their identity. Prefetch is the number of upcoming groups each reader prepares ahead. A bag is one observation’s group of items, which can have a different length from another group. Whole bags keeps those items together with their desired outcome; even an empty bag keeps its desired outcome."
        }
        return "**Goal**\n"+meaning+"\n\n**Build**\nAdd Dataset and DataLoader. Connect Dataset out to Train dataset and DataLoader out to Train loader. The checker supplies the model and source.\n\n**Configure**\n"+controlInstructions+" Apply your changes.\n\n**Check**\nChoose Check construction. Compare the measured row IDs, desired outcomes and grouping checks. Private assessment keeps its test rows and reference results hidden."
    }
    public var componentIDs:[String] {["dataset_source","data_loader"]}
    public var datasetValues:[String:JSONValue] {
        var values=DatasetConfiguration.defaults
        switch self {case .tensor:values["kind"] = .string("tensor");case .subset:values["kind"] = .string("subset");values["indices"] = .array([4,1,0]);case .iterable:values["kind"] = .string("boundedIterable");values["max_rows"]=5;default:break}
        return values
    }
    public var loaderValues:[String:JSONValue] {
        var values=DataLoaderConfiguration.defaults;values["batch_size"]=4;values["collation"] = .string("fixed_stack")
        if self == .shuffle {values["batch_size"]=3;values["shuffle"]=true;values["collation"] = .string("auto")}
        if self == .collation {values["collation"] = .string("padded_sequence")}
        if self == .continuation {values["shuffle"]=true;values["collation"] = .string("whole_bags");values["num_workers"]=2;values["prefetch_factor"]=1;values["timeout"] = .number(5)}
        return values
    }
}
public enum DataLoadingExamples {
    public static func lessonGraph(task:DataLoadingLearningTask)throws->GraphDocument {
        let dataset=GraphNode(id:"observations",type:"dataset_source",name:"Access training observations",x:30,y:60,config:try DatasetConfiguration(values:task.datasetValues).values)
        let loader=GraphNode(id:"groups",type:"data_loader",name:"Arrange training groups",x:30,y:240,config:try DataLoaderConfiguration(values:task.loaderValues).values)
        let train=GraphNode(id:"train",type:"train",name:"Supplied model and source",x:310,y:150,config:[:])
        return .init(name:task.title,nodes:[dataset,loader,train],edges:[.init(id:"source-recipe",source:dataset.id,sourcePort:"out",target:train.id,targetPort:"dataset"),.init(id:"batch-recipe",source:loader.id,sourcePort:"out",target:train.id,targetPort:"loader")])
    }
}
