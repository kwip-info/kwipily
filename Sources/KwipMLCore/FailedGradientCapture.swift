import Foundation

/// Separate partial-failure contract. Completed gradient v1 decoding is unchanged.
public struct FailedGradientCaptureContext: Codable, Equatable {
    public var workerRunID:String
    public var device:String
    public var initialEpoch:Int
    public var requestedFinalEpoch:Int
    public var requestedBatchSize:Int
    public var trainingRows:Int
    public var configuration:GradientCaptureConfiguration
    public var inventory:GradientCaptureInventory
}
public struct InterruptedGradientEpoch: Codable, Equatable {
    public var event:String
    public var schemaVersion:Int
    public var id:String
    public var runID:String
    public var epoch:Int
    public var stage:String
    public var lastBackwardBatchIndex:Int?
    public var totalBatches:Int
    public var totalTrainingRows:Int
    public var scheduledBatchCount:Int
    public var observedBatchCount:Int
    public var observedTrainingRows:Int
    public var captureBudgetRemaining:Int
    public var batches:[GradientBatchSummary]
}
public enum FailedGradientCaptureCodec {
    public static let identifier="kwip.failed-gradient-observations.v1"
    public static let maximumEventBytes=100_000
    public static let maximumArtifactBytes=8_000_000
    private static let batchKeys:Set<String>=["batchIndex","trainingRows","presentParameterCount","presentElementCount","missingParameterCount","missingElementCount","finiteElementCount","nonfiniteElementCount","zeroElementCount","l2Norm","maxAbs"]
    private static func require(_ value:Bool,_ message:String)throws {if !value {throw GradientCaptureError(message:message)}}
    private static func object(_ value:JSONValue,keys:Set<String>)throws->[String:JSONValue] {
        guard case .object(let map)=value,Set(map.keys)==keys else {throw GradientCaptureError(message:"Unknown or missing failed-gradient fields.")};return map
    }
    private static func read(_ data:Data,limit:Int)throws->JSONValue {
        try require(!data.isEmpty && data.count<=limit,"Failed-gradient evidence exceeds byte bounds.")
        return try JSONDecoder().decode(JSONValue.self,from:data)
    }
    public static func decodeContext(_ data:Data,graph:GraphDocument)throws->FailedGradientCaptureContext {
        let map=try object(read(data,limit:200_000),keys:["workerRunID","device","initialEpoch","requestedFinalEpoch","requestedBatchSize","trainingRows","configuration","inventory"])
        let inventory=try object(map["inventory"]!,keys:["parameterCount","elementCount","parameters"])
        guard case .array(let parameters)=inventory["parameters"] else {throw GradientCaptureError(message:"Missing parameter inventory.")}
        for p in parameters {_ = try object(p,keys:["name","nodeID","shape","elementCount"])}
        let value=try JSONDecoder().decode(FailedGradientCaptureContext.self,from:data)
        try validate(value,graph:graph);return value
    }
    public static func validate(_ context:FailedGradientCaptureContext,graph:GraphDocument)throws {
        try context.configuration.validate()
        try require(UUID(uuidString:context.workerRunID) != nil && ["cpu","mps"].contains(context.device),"Failed-gradient worker identity/device differs.")
        try require((0..<10000).contains(context.initialEpoch) && context.requestedFinalEpoch>context.initialEpoch && context.requestedFinalEpoch<=10000 && (1...100000).contains(context.requestedBatchSize) && (1...100000).contains(context.trainingRows),"Invalid requested failed-gradient coverage.")
        var expected:[String:[Int]]=[:]
        let parameterless:Set<String>=["input","output","relu","tanh","gelu","silu","leaky_relu","elu","softmax","add","concat","reshape","transpose","flatten","matmul","max_pool1d","max_pool2d","max_pool3d","avg_pool1d","avg_pool2d","avg_pool3d"]
        for node in graph.nodes {
            func dimension(_ name:String)throws->Int {
                guard let n=node.config[name]?.integerValue,(1...4096).contains(n) else {throw GradientCaptureError(message:"Invalid captured parameter dimensions.")};return n
            }
            let shape:[Int]
            if node.type=="linear" {shape=[try dimension("out_features"),try dimension("in_features")]}
            else if node.type=="bilinear" {shape=[try dimension("out_features"),try dimension("in1_features"),try dimension("in2_features")]}
            else {try require(parameterless.contains(node.type) || AdaptivePoolingConfiguration.componentIDs.contains(node.type),"Failed-gradient inventory verification currently supports Linear/Bilinear parameters only.");continue}
            try require(!node.id.isEmpty && node.id.count<=128 && !node.id.contains("."),"Unsupported captured parameter name.")
            expected["layers.\(node.id).weight"]=shape
            if node.config["bias"] != .bool(false) {expected["layers.\(node.id).bias"]=[shape[0]]}
        }
        let inventory=context.inventory
        try require((1...256).contains(inventory.parameterCount) && inventory.parameterCount==inventory.parameters.count && inventory.parameterCount==expected.count && (1...2_000_000).contains(inventory.elementCount),"Failed-gradient inventory bounds differ.")
        var names=Set<String>(),total=0
        for parameter in inventory.parameters {
            try require(names.insert(parameter.name).inserted && parameter.shape==expected[parameter.name] && parameter.name.hasPrefix("layers.\(parameter.nodeID).") && graph.nodes.contains(where:{$0.id==parameter.nodeID}),"Failed-gradient parameter differs from captured graph.")
            let count=parameter.shape.reduce(1,*)
            try require(count==parameter.elementCount && count<=2_000_000,"Failed-gradient parameter count differs.");total+=count
        }
        try require(total==inventory.elementCount,"Failed-gradient inventory total differs.")
    }
    private static func shapeBatches(_ map:[String:JSONValue])throws {
        guard case .array(let batches)=map["batches"],batches.count<=8 else {throw GradientCaptureError(message:"Invalid failed-gradient batch array.")}
        for b in batches {_ = try object(b,keys:batchKeys)}
    }
    public static func decodeCompleted(_ data:Data,context:FailedGradientCaptureContext,previous:[GradientEpochSummary])throws->GradientEpochSummary {
        let map=try object(read(data,limit:maximumEventBytes),keys:["event","schemaVersion","runID","epoch","totalBatches","scheduledBatchCount","observedBatchCount","totalTrainingRows","observedTrainingRows","captureBudgetRemaining","status","batches"])
        try shapeBatches(map)
        let value=try JSONDecoder().decode(GradientEpochSummary.self,from:data)
        try require(value.event=="gradientSummary" && value.schemaVersion==1 && value.runID==context.workerRunID,"Completed gradient prefix identity differs.")
        try coverage(epoch:value.epoch,totalBatches:value.totalBatches,totalRows:value.totalTrainingRows,scheduled:value.scheduledBatchCount,observed:value.observedBatchCount,observedRows:value.observedTrainingRows,remaining:value.captureBudgetRemaining,batches:value.batches,lastBackward:value.totalBatches-1,context:context,previous:previous)
        let status:GradientCaptureStatus=value.scheduledBatchCount==0 ? .notScheduled:(value.observedBatchCount==0 ? .budgetExhausted:.captured)
        try require(value.status==status,"Completed gradient prefix status differs.");return value
    }
    public static func decodeInterrupted(_ data:Data,context:FailedGradientCaptureContext,previous:[GradientEpochSummary])throws->InterruptedGradientEpoch {
        let map=try object(read(data,limit:maximumEventBytes),keys:["event","schemaVersion","id","runID","epoch","stage","lastBackwardBatchIndex","totalBatches","totalTrainingRows","scheduledBatchCount","observedBatchCount","observedTrainingRows","captureBudgetRemaining","batches"])
        try shapeBatches(map)
        let value=try JSONDecoder().decode(InterruptedGradientEpoch.self,from:data)
        try require(value.event=="gradientInterrupted" && value.schemaVersion==1 && value.id==identifier && value.runID==context.workerRunID,"Interrupted gradient identity differs.")
        try require(["trainingForward","trainingBackward","optimizerStep","validationForward","validationLoss"].contains(value.stage),"Unknown failed-gradient failure stage.")
        if value.stage.hasPrefix("validation") {try require(value.lastBackwardBatchIndex==value.totalBatches-1,"Validation failure requires completed training backward boundary.")}
        if value.stage=="optimizerStep" {try require(value.lastBackwardBatchIndex != nil,"Optimizer failure requires an actual backward boundary.")}
        try coverage(epoch:value.epoch,totalBatches:value.totalBatches,totalRows:value.totalTrainingRows,scheduled:value.scheduledBatchCount,observed:value.observedBatchCount,observedRows:value.observedTrainingRows,remaining:value.captureBudgetRemaining,batches:value.batches,lastBackward:value.lastBackwardBatchIndex,context:context,previous:previous)
        return value
    }
    private static func coverage(epoch:Int,totalBatches:Int,totalRows:Int,scheduled:Int,observed:Int,observedRows:Int,remaining:Int,batches:[GradientBatchSummary],lastBackward:Int?,context:FailedGradientCaptureContext,previous:[GradientEpochSummary])throws {
        try require(previous.count<10000 && epoch==context.initialEpoch+previous.count+1 && epoch<=context.requestedFinalEpoch,"Failed-gradient epoch sequence differs.")
        let size=context.requestedBatchSize,total=(context.trainingRows+size-1)/size
        try require(totalBatches==total && totalRows==context.trainingRows,"Failed-gradient row/batch counts differ.")
        if let lastBackward {try require((0..<total).contains(lastBackward),"Invalid interrupted backward boundary.")}
        let count=(epoch-1)%context.configuration.everyEpochs==0 ? min(context.configuration.maxBatchesPerEpoch,total):0
        let budget=previous.last?.captureBudgetRemaining ?? context.configuration.maxCaptures
        let indices=(0..<count).map {count==1 ? 0:$0*(total-1)/(count-1)}.filter {$0 <= (lastBackward ?? -1)}
        let expected=Array(indices.prefix(budget))
        try require(scheduled==count && observed==expected.count && batches.count==observed && remaining==budget-observed,"Failed-gradient sample cadence or budget differs.")
        var rows=0
        for (index,b) in batches.enumerated() {
            try require(b.batchIndex==expected[index] && b.trainingRows==min(size,context.trainingRows-b.batchIndex*size),"Failed-gradient sampled minibatch rows differ.")
            let pc=[b.presentParameterCount,b.missingParameterCount],ec=[b.presentElementCount,b.missingElementCount,b.finiteElementCount,b.nonfiniteElementCount,b.zeroElementCount]
            try require(pc.allSatisfy{(0...context.inventory.parameterCount).contains($0)} && ec.allSatisfy{(0...context.inventory.elementCount).contains($0)},"Failed-gradient counts exceed inventory.")
            try require(b.presentParameterCount+b.missingParameterCount==context.inventory.parameterCount && b.presentElementCount+b.missingElementCount==context.inventory.elementCount && b.finiteElementCount+b.nonfiniteElementCount==b.presentElementCount && b.zeroElementCount<=b.finiteElementCount && b.presentElementCount>=b.presentParameterCount && b.missingElementCount>=b.missingParameterCount && (b.presentParameterCount==0)==(b.presentElementCount==0) && (b.missingParameterCount==0)==(b.missingElementCount==0),"Failed-gradient element identities differ.")
            if b.presentElementCount==0 || b.nonfiniteElementCount>0 {try require(b.l2Norm==nil && b.maxAbs==nil,"Missing/nonfinite gradients require null norms.")}
            else {
                guard let l2=b.l2Norm,let maximum=b.maxAbs else {throw GradientCaptureError(message:"Finite observed gradients need norms.")}
                try require(l2.isFinite && maximum.isFinite && maximum>=0 && l2>=maximum && l2<=maximum*sqrt(Double(b.presentElementCount))*(1+1e-12) && (maximum==0)==(b.zeroElementCount==b.finiteElementCount),"Invalid failed-gradient norm.")
            }
            rows+=b.trainingRows
        }
        try require(rows==observedRows && rows<=context.trainingRows,"Failed-gradient observed rows differ.")
    }
}
