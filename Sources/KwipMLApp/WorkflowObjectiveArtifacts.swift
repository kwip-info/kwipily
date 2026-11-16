import Foundation
import KwipMLCore

struct WorkflowObjectiveArtifactContext {
    let recipe:LossRecipe
    let runID:String
    let inputContract:TypedTensorContract
    let outputSampleShape:[Int]
    let trainingCount:Int
    let evaluationCount:Int
    let expectedEpochs:Int
    let scalarization:WorkflowObjectiveScalarization
    let device:ContractDevice
    let policy:TrainingSampleCapturePolicy
    let explicitSampleCapture:Bool
    let namedInputPlan:NamedInputPlan?
    let usesNamedValues:Bool
    init(recipe:LossRecipe,runID:String,inputContract:TypedTensorContract,outputSampleShape:[Int],trainingCount:Int,evaluationCount:Int,expectedEpochs:Int,scalarization:WorkflowObjectiveScalarization,device:ContractDevice,policy:TrainingSampleCapturePolicy,explicitSampleCapture:Bool=false,namedInputPlan:NamedInputPlan?=nil,usesNamedValues:Bool=false)throws {
        self.namedInputPlan=namedInputPlan;self.usesNamedValues=usesNamedValues
        try recipe.configuration.validate(targetMapping:recipe.targetMapping)
        guard !runID.isEmpty,(1...100_000_000).contains(trainingCount),(1...100_000_000).contains(evaluationCount),(1...10_000).contains(expectedEpochs),[ContractDevice.cpu,.mps].contains(device),[ContractDType.float32,.int64].contains(inputContract.dtype),(1...7).contains(outputSampleShape.count) else {throw WorkflowPhaseError("Objective result context has invalid source, partition, device or geometry.")}
        var elements=1
        for d in outputSampleShape {guard (1...4096).contains(d),elements<=100_000_000/d else {throw WorkflowPhaseError("Objective produced shape exceeds its element budget.")};elements *= d}
        self.recipe=recipe;self.runID=runID;self.inputContract=inputContract;self.outputSampleShape=outputSampleShape
        self.trainingCount=trainingCount;self.evaluationCount=evaluationCount;self.expectedEpochs=expectedEpochs
        self.scalarization=scalarization;self.device=device;self.policy=policy;self.explicitSampleCapture=explicitSampleCapture
    }
    func evaluationContext() throws -> WorkflowObjectiveEvaluationContext {
        try .init(recipe:recipe,runID:runID,outputSampleShape:outputSampleShape,
                  trainingCount:trainingCount,evaluationCount:evaluationCount,expectedEpochs:expectedEpochs,
                  scalarization:scalarization,device:device,policy:policy,explicitSampleCapture:explicitSampleCapture)
    }
}
enum ObjectiveCaptureStatus:String,Equatable {case captured,omittedByPolicy}
struct ObjectivePredictionRow:Equatable {
    let input:ObjectiveResultTensor
    let output:ObjectiveResultTensor
    let targets:[String:ObjectiveResultTensor]
    let objectiveOutput:ObjectiveResultTensor?
    let decoded:ObjectiveResultTensor?
    var namedInputs:[String:ObjectiveResultTensor]? = nil
}
struct ObjectivePredictionPreview:Equatable {
    let runID:String
    let source:WorkflowNodeAddress
    let evaluationCount:Int
    let rows:[ObjectivePredictionRow]
    let captureStatus:ObjectiveCaptureStatus
}
struct ObjectiveRawPreview:Equatable {
    let shape:[Int]
    let dtype:ContractDType
    let values:[Double]
    let captureStatus:ObjectiveCaptureStatus
}
struct ObjectiveEvaluation:Equatable {
    var sequenceObjective=false
    var perplexity:Double? {guard sequenceObjective,effectiveScalar>=0,effectiveScalar<log(Double.greatestFiniteMagnitude) else {return nil};return exp(effectiveScalar)}
    let componentID:String
    let observations:Int
    let activeCount:Int
    let ignoredCount:Int
    let maskedCount:Int
    let effectiveScalar:Double
    let aggregationNumerator:Double
    let aggregationDenominator:Double
    let aggregationConvention:String
    let rawShape:[Int]
    let accuracy:Double?
    let rawPreview:ObjectiveRawPreview?
}
struct ObjectiveEpochMetric:Equatable,Identifiable {
    let epoch:Int
    let trainLoss:Double
    let validationLoss:Double
    let accuracy:Double?
    let validation:ObjectiveEvaluation
    var id:Int {epoch}
}
enum WorkflowObjectiveArtifacts {
    // Dense callers retain their original entry points. Only arithmetic context is shared.
    static func evaluation(_ value: JSONValue, context: WorkflowObjectiveArtifactContext) throws -> ObjectiveEvaluation {
        try evaluation(value,context:context.evaluationContext())
    }
    static func epoch(_ value: JSONValue, expectedEpoch: Int, context: WorkflowObjectiveArtifactContext) throws -> ObjectiveEpochMetric {
        try epoch(value,expectedEpoch:expectedEpoch,context:context.evaluationContext())
    }
    static func metrics(_ data: Data, context: WorkflowObjectiveArtifactContext) throws -> [ObjectiveEpochMetric] {
        try metrics(data,context:context.evaluationContext())
    }
    private static func fail<T>(_ text:String)throws->T {throw WorkflowPhaseError(text)}
    private static func object(_ value:JSONValue)throws->[String:JSONValue] {guard case .object(let result)=value else {return try fail("Objective record must be an object.")};return result}
    private static func number(_ value:JSONValue?)throws->Double {guard let n=value?.doubleValue,n.isFinite else {return try fail("Objective numeric evidence is missing or nonfinite.")};return n}
    private static func integer(_ value:JSONValue?,minimum:Int=0)throws->Int {guard let n=value?.integerValue,n>=minimum else {return try fail("Objective count is missing or invalid.")};return n}
    private static func shape(_ value:JSONValue?)throws->[Int] {guard case .array(let array)=value else {return try fail("Objective shape is missing.")};return try array.map {try integer($0,minimum:1)}}
    private static func near(_ a:Double,_ b:Double)->Bool {abs(a-b)<=max(1e-6,max(abs(a),abs(b))*1e-5)}
    private static func decode(_ data:Data,limit:Int)throws->JSONValue {guard !data.isEmpty,data.count<=limit else {return try fail("Objective artifact exceeds its byte limit.")};return try JSONDecoder().decode(JSONValue.self,from:data)}
    private static func rawSample(_ c:WorkflowObjectiveEvaluationContext)throws->[Int] {
        let op=c.recipe.configuration.componentID,v=c.recipe.configuration.values,out=c.outputSampleShape
        if ["loss_cross_entropy","loss_nll"].contains(op) {
            let axis=v["class_axis"]!.integerValue!,rank=out.count+1,actual=axis<0 ? rank+axis:axis
            guard actual>0,actual<rank else {return try fail("Class direction is outside the produced values.")}
            var result=out;result.remove(at:actual-1);return result
        }
        if ["loss_ctc","loss_triplet_margin"].contains(op) {return []}
        if ["loss_mse","loss_l1"].contains(op),let target=c.recipe.targetMapping.roles.first(where:{$0.role == "target"}) {
            let rank=max(out.count,target.sampleShape.count),a=Array(repeating:1,count:rank-out.count)+out,b=Array(repeating:1,count:rank-target.sampleShape.count)+target.sampleShape
            guard zip(a,b).allSatisfy({$0 == $1 || $0 == 1 || $1 == 1}) else {return try fail("Desired and produced shapes cannot share directions.")}
            return zip(a,b).map {max($0,$1)}
        }
        return out
    }
    private static func classificationEligible(_ recipe:LossRecipe)->Bool {
        ["loss_cross_entropy","loss_nll"].contains(recipe.configuration.componentID) && recipe.targetMapping.roles.first(where:{$0.role == "target"})?.dtype == .int64
    }
    static func predictions(_ data:Data,metadata:JSONValue,context c:WorkflowObjectiveArtifactContext)throws->ObjectivePredictionPreview {
        let m=try object(metadata)
        let schema=c.usesNamedValues ? 4:c.explicitSampleCapture ? 2:1
        guard Set(m.keys) == ["schemaVersion","file","sha256","bytes"],m["schemaVersion"]?.integerValue == schema,m["file"]?.stringValue == "predictions.json",m["bytes"]?.integerValue == data.count,m["sha256"]?.stringValue == WorkflowPhaseArtifacts.hash(data) else {return try fail("Prediction artifact differs from its captured byte identity.")}
        let p=try object(decode(data,limit:1_000_000))
        let keys:Set<String>=["schemaVersion","runID","selection","evaluationCount","predictionInput","predictionRows"]
        guard Set(p.keys) == (c.explicitSampleCapture || c.usesNamedValues ? keys.union(["captureStatus"]):keys),p["captureStatus"] == (c.explicitSampleCapture || c.usesNamedValues ? .string(c.policy.predictionRows ? "captured":"omittedByPolicy"):nil),p["schemaVersion"]?.integerValue == schema,p["runID"]?.stringValue == c.runID,p["selection"]?.stringValue == "firstRowsOfValidationPartition",p["evaluationCount"]?.integerValue == c.evaluationCount else {return try fail("Prediction artifact identity or selection differs.")}
        let input=try object(p["predictionInput"]!)
        if c.usesNamedValues {
            guard let plan=c.namedInputPlan,case .object(let declarations)=input["inputs"],input["kind"] == .string("namedInputs"),input["schemaVersion"] == 1,Set(declarations.keys)==Set(plan.inputs.map(\.name)) else {return try fail("Named prediction contract differs.")}
            for entry in plan.inputs {
                guard case .object(let declared)=declarations[entry.name],declared["nodeID"] == .string(entry.nodeID!),declared["dtype"] == .string(entry.dtype),declared["sampleShape"] == .array(entry.sampleShape.map {.integer(Int64($0))}) else {return try fail("Named prediction geometry differs.")}
            }
        } else {guard input["dtype"]?.stringValue == c.inputContract.dtype.rawValue,try shape(input["sampleShape"]) == c.inputContract.layout.sample.dimensions else {return try fail("Prediction input geometry differs.")}}
        let admittedRows = c.policy.predictionRows ? try c.namedInputPlan.map {try SequenceTrainingContract.predictionRowLimit(plan:$0,outputShape:c.outputSampleShape,count:c.evaluationCount)} ?? min(200,c.evaluationCount):0
        guard case .array(let rows)=p["predictionRows"]!,rows.count == admittedRows else {return try fail("Prediction capture permissions differ.")}
        let reduction=c.recipe.configuration.values["reduction"]!.stringValue!,raw=try rawSample(c.evaluationContext()),decodedShape=raw
        let parsed=try rows.map {value->ObjectivePredictionRow in
            let row=try object(value)
            var keys:Set<String>=["input","output","targets"]
            if reduction == "none" {keys.insert("objectiveOutput")}
            if classificationEligible(c.recipe) {keys.insert("decoded")}
            guard Set(row.keys)==keys else {return try fail("Prediction row has missing or unknown objective values.")}
            let targets=try object(row["targets"]!)
            guard Set(targets.keys)==Set(c.recipe.targetMapping.roles.map(\.role)) else {return try fail("Prediction desired-value roles differ from the captured mapping.")}
            var tensors:[String:ObjectiveResultTensor]=[:]
            for role in c.recipe.targetMapping.roles {tensors[role.role]=try ObjectiveResultTensor(json:targets[role.role]!,shape:role.sampleShape,dtype:role.dtype)}
            var named:[String:ObjectiveResultTensor]?=nil
            let source:ObjectiveResultTensor
            if c.usesNamedValues {
                guard let plan=c.namedInputPlan,case .object(let values)=row["input"],Set(values.keys)==Set(plan.inputs.map(\.name)) else {return try fail("Named prediction row differs from the captured inputs.")}
                var decoded:[String:ObjectiveResultTensor]=[:]
                for entry in plan.inputs {decoded[entry.name]=try ObjectiveResultTensor(json:values[entry.name]!,shape:entry.sampleShape,dtype:ContractDType(rawValue:entry.dtype)!)}
                named=decoded;source=decoded[plan.inputs[0].name]!
            } else {source=try ObjectiveResultTensor(json:row["input"]!,shape:c.inputContract.layout.sample.dimensions,dtype:c.inputContract.dtype)}
            return try ObjectivePredictionRow(input:source,output:ObjectiveResultTensor(json:row["output"]!,shape:c.outputSampleShape,dtype:.float32),targets:tensors,objectiveOutput:row["objectiveOutput"].map {try ObjectiveResultTensor(json:$0,shape:raw,dtype:.float32)},decoded:row["decoded"].map {try ObjectiveResultTensor(json:$0,shape:decodedShape,dtype:.int64)},namedInputs:named)
        }
        return .init(runID:c.runID,source:c.recipe.source,evaluationCount:c.evaluationCount,rows:parsed,captureStatus:c.policy.predictionRows ? .captured:.omittedByPolicy)
    }
    static func evaluation(_ value:JSONValue,context c:WorkflowObjectiveEvaluationContext)throws->ObjectiveEvaluation {
        let p=try object(value),v=c.recipe.configuration.values,reduction=v["reduction"]!.stringValue!
        let expectedRaw=reduction == "none" ? [c.evaluationCount]+(try rawSample(c)):[]
        guard p["componentID"]?.stringValue == c.recipe.configuration.componentID,p["reduction"]?.stringValue == reduction,p["scalarization"] == (reduction == "none" ? .string(c.scalarization.rawValue):.null),p["device"]?.stringValue == c.device.rawValue,try integer(p["observations"],minimum:1) == c.evaluationCount,try shape(p["rawShape"]) == expectedRaw else {return try fail("Objective evaluation contract differs from its captured plan.")}
        let active=try integer(p["activeCount"]),ignored=try integer(p["ignoredCount"]),masked=try integer(p["maskedCount"]),score=try number(p["effectiveScalar"]),numerator=try number(p["aggregationNumerator"]),denominator=try number(p["aggregationDenominator"])
        let sum=reduction == "sum" || reduction == "none" && c.scalarization == .sum
        let convention=sum ? "sum":reduction == "none" ? "rawElementMean":reduction == "batchmean" ? "nativeBatchMean":"nativeMean"
        guard denominator>0,near(score,numerator/denominator),p["aggregationConvention"]?.stringValue == convention,!sum || denominator == 1 else {return try fail("Objective aggregate arithmetic or convention differs.")}
        var positions=c.evaluationCount
        if ["loss_cross_entropy","loss_nll"].contains(c.recipe.configuration.componentID) {for d in try rawSample(c) {guard positions<=100_000_000/d else {return try fail("Objective position count exceeds budget.")};positions *= d}}
        guard active<=positions,ignored<=positions,masked<=positions,active+ignored+masked == positions else {return try fail("Objective active, ignored and masked counts do not fill the partition.")}
        var accuracy:Double?
        if let classification=p["classification"] {
            guard classificationEligible(c.recipe) else {return try fail("Accuracy is unavailable for this objective.")}
            let cl=try object(classification),eligible=try integer(cl["eligibleCount"]),correct=try integer(cl["correctCount"])
            guard cl["rule"]?.stringValue == "argmax",cl["classAxis"] == v["class_axis"],cl["ignoreIndex"] == v["ignore_index"],eligible == active,correct<=eligible else {return try fail("Classification counts or rule differ.")}
            if eligible>0 {let a=try number(cl["accuracy"]);guard near(a,Double(correct)/Double(eligible)) else {return try fail("Classification accuracy differs from its counts.")};accuracy=a}
            else {guard cl["accuracy"] == .null else {return try fail("Empty classification cannot claim accuracy.")}}
        } else if classificationEligible(c.recipe) {return try fail("Classification evidence is missing.")}
        var preview:ObjectiveRawPreview?
        if let raw=p["rawPreview"] {
            let r=try object(raw)
            guard try shape(r["shape"]) == expectedRaw,r["dtype"]?.stringValue == "float32",case .array(let values)=r["values"],values.count<=8,r["captureStatus"]?.stringValue == (c.policy.activationValues ? "captured":"omittedByPolicy"),c.policy.activationValues || values.isEmpty else {return try fail("Raw objective preview violates its geometry or independent capture permission.")}
            var total=1;for d in expectedRaw {guard total<=100_000_000/d else {return try fail("Raw preview shape exceeds budget.")};total *= d}
            guard values.count == (c.policy.activationValues ? min(8,total):0) else {return try fail("Raw preview prefix is incomplete.")}
            let parsed=try values.map {try ObjectiveResultTensor(json:$0,shape:[],dtype:.float32).values[0]}
            preview = .init(shape:expectedRaw,dtype:.float32,values:parsed,captureStatus:c.policy.activationValues ? .captured:.omittedByPolicy)
        }
        return .init(sequenceObjective:c.recipe.targetMapping.schemaVersion==2,componentID:c.recipe.configuration.componentID,observations:c.evaluationCount,activeCount:active,ignoredCount:ignored,maskedCount:masked,effectiveScalar:score,aggregationNumerator:numerator,aggregationDenominator:denominator,aggregationConvention:convention,rawShape:expectedRaw,accuracy:accuracy,rawPreview:preview)
    }
    static func epoch(_ value:JSONValue,expectedEpoch:Int,context c:WorkflowObjectiveEvaluationContext)throws->ObjectiveEpochMetric {
        guard (1...c.expectedEpochs).contains(expectedEpoch) else {return try fail("Objective epoch is outside the captured history.")}
            let p=try object(value),objective=try object(p["objective"] ?? .null),reduction=c.recipe.configuration.values["reduction"]!.stringValue!
            guard p["event"]?.stringValue == "epoch",p["epoch"]?.integerValue == expectedEpoch,objective["componentID"]?.stringValue == c.recipe.configuration.componentID,objective["reduction"]?.stringValue == reduction,objective["scalarization"] == (reduction == "none" ? .string(c.scalarization.rawValue):.null) else {return try fail("Epoch identity or objective recipe differs.")}
            let validation=try evaluation(objective["validation"] ?? .null,context:c),train=try number(p["trainLoss"]),loss=try number(p["validationLoss"]),training=try object(objective["training"] ?? .null),numerator=try number(training["aggregationNumerator"]),denominator=try number(training["aggregationDenominator"])
            guard try integer(training["observations"],minimum:1) == c.trainingCount,denominator>0,near(train,numerator/denominator),near(loss,validation.effectiveScalar) else {return try fail("Epoch partition aggregation differs.")}
            let active=try integer(training["activeCount"]),ignored=try integer(training["ignoredCount"]),masked=try integer(training["maskedCount"])
            var positions=c.trainingCount
            if ["loss_cross_entropy","loss_nll"].contains(c.recipe.configuration.componentID) {for d in try rawSample(c) {guard positions<=100_000_000/d else {return try fail("Training position count exceeds budget.")};positions *= d}}
            guard active<=positions,ignored<=positions,masked<=positions,active+ignored+masked == positions else {return try fail("Training active, ignored and masked counts do not fill the partition.")}
            let accuracy=p["accuracy"].map {try? number($0)} ?? nil
            guard accuracy == nil ? validation.accuracy == nil:classificationEligible(c.recipe) && near(accuracy!,validation.accuracy ?? -1),p["accuracy"] == nil || accuracy != nil else {return try fail("Epoch accuracy is missing, invalid or unsupported.")}
            return ObjectiveEpochMetric(epoch:expectedEpoch,trainLoss:train,validationLoss:loss,accuracy:accuracy,validation:validation)
    }
    static func metrics(_ data:Data,context c:WorkflowObjectiveEvaluationContext)throws->[ObjectiveEpochMetric] {
        guard case .array(let rows)=try decode(data,limit:4_000_000),rows.count == c.expectedEpochs,rows.count<=10_000 else {return try fail("Objective epoch history is incomplete or exceeds its limit.")}
        return try rows.enumerated().map {try epoch($0.element,expectedEpoch:$0.offset+1,context:c)}
    }
}
