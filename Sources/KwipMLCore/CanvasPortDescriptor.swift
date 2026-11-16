import Foundation

/// Presentation only: connection validation and conversions remain owned by execution contracts.
public enum CanvasPortKind:String,Equatable { case tensor,table,dataset,fittedState,modelDefinition,trainingData,runConfiguration,frozenModel,optimizerRecipe,schedulerRecipe,objectiveRecipe,initializationRecipe,gradientPolicyRecipe,trainabilityRecipe,datasetRecipe,batchRecipe,unknown }
public enum CanvasPortGlyph:String,Equatable { case circle,square,stack,diamond,document,mapping,sliders,lock,question }
public struct CanvasPortDescriptor:Equatable {
    public var kind:CanvasPortKind
    public var direction:PortDirection
    public var role:String?
    public var partition:PartitionRole?
    public var declared:[String]
    /// Reserved for explicitly verified evidence. Registry projections never populate this field.
    public var measured:[String]
    public init(kind:CanvasPortKind,direction:PortDirection,role:String?=nil,partition:PartitionRole?=nil,declared:[String]=[],measured:[String]=[]) {
        self.kind=kind;self.direction=direction;self.role=role;self.partition=partition;self.declared=declared;self.measured=measured
    }
    public var glyph:CanvasPortGlyph { switch kind { case .tensor:return .circle;case .table:return .square;case .dataset:return .stack;case .fittedState:return .diamond;case .modelDefinition:return .document;case .trainingData:return .mapping;case .runConfiguration:return .sliders;case .datasetRecipe,.batchRecipe,.optimizerRecipe,.schedulerRecipe,.objectiveRecipe,.initializationRecipe,.gradientPolicyRecipe,.trainabilityRecipe:return .sliders;case .frozenModel:return .lock;case .unknown:return .question } }
    public var shortLabel:String { switch kind {case .tensor:return "Tensor";case .table:return "Table";case .dataset:return "Dataset";case .fittedState:return "Fitted state";case .modelDefinition:return "Model definition";case .trainingData:return "Training data mapping";case .runConfiguration:return "Run configuration";case .optimizerRecipe:return "Optimizer recipe";case .schedulerRecipe:return "Scheduler recipe";case .objectiveRecipe:return "Objective recipe";case .initializationRecipe:return "Starting settings recipe";case .gradientPolicyRecipe:return "Gradient policy recipe";case .trainabilityRecipe:return "Trainability recipe";case .datasetRecipe:return "Dataset view recipe";case .batchRecipe:return "Batch reader recipe";case .frozenModel:return "Trained model";case .unknown:return "Unknown type"} }
    public var accessibilityDescription:String {
        ([direction == .input ? "Input":"Output",shortLabel]+[role,partition.map { $0.rawValue + " partition" }].compactMap {$0}+(declared.isEmpty ? []:["Declared: " + declared.joined(separator:", ")])+(measured.isEmpty ? []:["Measured: " + measured.joined(separator:", ")])).joined(separator:" · ")
    }
    public static func unknown(direction:PortDirection,reason:String="Port contract unavailable")->Self { .init(kind:.unknown,direction:direction,declared:[reason]) }
    public static func contract(_ port:PortContract,configuration:[String:JSONValue]=[:])->Self {
        let payloadCount=[port.schema.tensor != nil,port.schema.table != nil,port.schema.dataset != nil,port.schema.estimator != nil,port.schema.state != nil,port.schema.artifact != nil].filter {$0}.count
        guard payloadCount == 1 else {return .unknown(direction:port.direction,reason:"Malformed port declaration")}
        switch port.schema.kind {
        case .tensor:
            guard let tensor=port.schema.tensor else {return .unknown(direction:port.direction)}
            let dimensions=tensor.dimensions.map { dimension -> String in
                switch dimension.kind {
                case .fixed:return dimension.value.map(String.init) ?? "unresolved"
                case .symbol:return dimension.name ?? "unresolved"
                case .parameter:
                    let name=dimension.name ?? "dimension"
                    if let value=configuration[name]?.doubleValue,value.isFinite,value>0,value.rounded()==value,value<1_000_000 {return String(Int(value))}
                    return name + " unresolved"
                }
            }
            let displayed=tensor.leadingAxes == nil ? dimensions:["…"]+dimensions
            var declared=["["+displayed.joined(separator:", ")+"]",tensor.dtype.rawValue]
            if let axes=tensor.leadingAxes {declared.append("Rank \(axes.minimum+dimensions.count)–\(axes.maximum+dimensions.count); shape inferred from connections")}
            return .init(kind:.tensor,direction:port.direction,declared:declared)
        case .table:
            guard let table=port.schema.table else {return .unknown(direction:port.direction)}
            return .init(kind:.table,direction:port.direction,declared:table.columns.map {$0.name+": "+$0.dtype.rawValue})
        case .dataset:
            guard let dataset=port.schema.dataset else {return .unknown(direction:port.direction)}
            return .init(kind:.dataset,direction:port.direction,partition:dataset.partition)
        case .state:
            if port.schema.state?.kind == .scheduler {return .init(kind:.schedulerRecipe,direction:port.direction,role:"Scheduler recipe")}
            if port.schema.state?.kind == .optimizer {return .init(kind:.optimizerRecipe,direction:port.direction,role:"Optimizer recipe")}
            guard port.schema.state?.kind == .fittedTransform else {return .unknown(direction:port.direction,reason:"State kind has no reviewed visual mapping")}
            return .init(kind:.fittedState,direction:port.direction,role:"Fitted transform")
        default:return .unknown(direction:port.direction,reason:"Contract kind has no reviewed visual mapping")
        }
    }
    public static func data(_ port:DataFlowOperatorPort,direction:PortDirection)->Self {
        let kind:CanvasPortKind
        switch port.kind {case "tensor":kind = .tensor;case "table":kind = .table;case "dataset":kind = .dataset;case "state" where port.stateKind == "fittedTransform":kind = .fittedState;default:kind = .unknown}
        return .init(kind:kind,direction:direction,role:kind == .fittedState ? "Fitted transform":nil,partition:port.partition.flatMap(PartitionRole.init(rawValue:)),declared:kind == .unknown ? ["Unrecognized registry kind: " + port.kind]:[])
    }
}
