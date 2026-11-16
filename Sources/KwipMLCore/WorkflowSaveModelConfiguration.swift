import Foundation

/// Existing numeric packages retain their default; learned preparation is an explicit Prebuilt package option.
public struct WorkflowSaveModelConfiguration:Equatable {
    public enum Format:String,Codable,CaseIterable {case package,checkpoint,huggingFace}
    public enum Content:String,CaseIterable {case full,adapter,merged,continuation}
    public enum Preparation:String,CaseIterable {case none,learned}
    public let preparation:Preparation
    public let content:Content
    public let format:Format
    public let shardBytes:Int
    public init(format:Format = .package,shardBytes:Int = 2_000_000_000,content:Content = .full,preparation:Preparation = .none) {self.format=format;self.shardBytes=shardBytes;self.content=content;self.preparation=preparation}
    /// A standalone LoRA result includes its merged base weights. Resolve the inherited
    /// default identically for the inspector and export, without rewriting saved projects.
    public func effectiveContent(isLoRA:Bool)->Content {
        format == .huggingFace && isLoRA && content == .full ? .merged:content
    }
    public var values:[String:JSONValue] {var result:[String:JSONValue]=["format":.string(format.rawValue)];if format == .huggingFace {result["shardBytes"] = .integer(Int64(shardBytes))};if content != .full {result["content"] = .string(content.rawValue)};if preparation != .none {result["preparation"] = .string(preparation.rawValue)};return result}
    public static var configurationSchema:JSONValue {.object(["type":.string("object"),"additionalProperties":false,"properties":.object(["preparation":.object(["type":.string("string"),"enum":.array(Preparation.allCases.map{.string($0.rawValue)}),"description":.string("Package only: learned includes verified model-bound Preprocess rules; none preserves prepared numeric inputs.")]),"format":.object(["enum":.array(Format.allCases.map{.string($0.rawValue)})]),"content":.object(["type":.string("string"),"enum":.array(Content.allCases.map{.string($0.rawValue)}),"description":.string("Hugging Face format only. Omit content entirely for package and checkpoint, including full; those formats retain their required model state automatically.")]),"shardBytes":.object(["type":.string("integer"),"minimum":1_048_576,"maximum":2_000_000_000,"description":.string("Hugging Face folder only; omit for package/checkpoint.")])])])}
    public static func decode(_ config:[String:JSONValue])throws->Self {
        guard Set(config.keys).isSubset(of:["format","shardBytes","content","preparation"]) else {throw DataFlowContractError("Unknown Save model setting. Export destinations are private bindings, not workflow source.")}
        guard let raw=config["format"] else {guard config.isEmpty else {throw DataFlowContractError("Choose a Save model format before its options.")};return .init()}
        guard case .string(let name)=raw,let format=Format(rawValue:name) else {throw DataFlowContractError("Save model format must be package, checkpoint or huggingFace.")}
        let shard:Int
        if let value=config["shardBytes"] {guard format == .huggingFace,case .integer(let bytes)=value,(1_048_576...2_000_000_000).contains(bytes) else {throw DataFlowContractError("Choose a Hugging Face shard limit between 1 MiB and 2 GB.")};shard=Int(bytes)} else {shard=2_000_000_000}
        let content:Content
        if let value=config["content"] {guard format == .huggingFace,let name=value.stringValue,let parsed=Content(rawValue:name) else {throw DataFlowContractError("Choose supported fine-tuned model contents.")};content=parsed} else {content = .full}
        let preparation:Preparation
        if let value=config["preparation"] {guard format == .package,let name=value.stringValue,let parsed=Preparation(rawValue:name) else {throw DataFlowContractError("Learned preparation is a Prebuilt package option.")};preparation=parsed} else {preparation = .none}
        return .init(format:format,shardBytes:shard,content:content,preparation:preparation)
    }
}
public struct WorkflowSaveModelPlan:Equatable {
    public let projectID:String
    public let revision:UInt64
    public let save:WorkflowNodeAddress
    public let modelSource:WorkflowTrainedModelSource
    public let configuration:WorkflowSaveModelConfiguration
    public static func compile(_ document:WorkflowDocument,save:WorkflowNodeAddress)throws->Self {
        try WorkflowCodec.validate(document)
        guard document.schemaVersion>=17,document.drafts.isEmpty,let node=document.nodes.first(where:{$0.address==save}),case .saveModel(let config)=node.payload else {throw workflowFailure(.invalidSelection,"Choose a current Save model without unresolved drafts.",save)}
        guard document.edges.filter({$0.to.address==save}).allSatisfy({$0.to.port=="model"}) else {throw workflowFailure(.invalidPort,"Save model only accepts its Trained model input.",save)}
        guard !document.edges.contains(where:{$0.from.address==save}) else {throw workflowFailure(.invalidPort,"Save model ends this artifact branch and has no output ports.",save)}
        return try .init(projectID:document.projectID,revision:document.revision,save:save,modelSource:WorkflowTrainedModelSource.resolve(in:document,consumer:save),configuration:WorkflowSaveModelConfiguration.decode(config))
    }
}
