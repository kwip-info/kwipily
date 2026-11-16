import Foundation
import KwipMLCore

/// Version four reads the captured objective and geometry; output width never selects a loss.
enum WorkflowObjectiveEvaluationArtifacts {
    struct Verified {
        let summary:WorkflowEvaluationSummary
        let rows:[WorkflowEvaluationObjectivePrediction]
        let evaluation:ObjectiveEvaluation
        let table:WorkflowPredictionTableEvidence
    }
    static func verify(manifest:WorkflowEvaluationManifest,contents:[String:Data],request:WorkflowEvaluationWorkerRequest,snapshot:WorkflowFrozenModelSnapshot)throws->Verified {
        func pinned(_ path:String)throws->Data {
            guard let file=snapshot.files.first(where:{$0.path==path}) else {throw failure("Frozen objective evidence is missing.")}
            let bytes=try DiagnosticFileIO.read(root:snapshot.directory,parts:WorkflowArchiveCodec.pathParts(path),limit:WorkflowFrozenModelSnapshot.fileLimit(path))
            guard file.bytes==bytes.count,WorkflowArchiveCodec.hash(bytes)==file.sha256 else {throw failure("Frozen objective evidence changed.")};return bytes
        }
        let capture=try WorkflowArchiveCodec.decodeCapture(pinned("capture.json"))
        let registry=try JSONDecoder().decode(ComponentRegistry.self,from:pinned("model-registry.json"))
        let trained=try JSONDecoder().decode([String:JSONValue].self,from:pinned("training/artifacts/manifest.json"))
        guard manifest.schemaVersion==4,let selected=capture.plan.lossRecipe,let scalarization=capture.plan.objectiveScalarization,
              case .object(let checkpoint)=trained["checkpoint"],case .object(let contract)=checkpoint["contract"],
              case .object(let settings)=trained["settings"],case .object(let provenance)=trained["provenance"],case .object(let runtime)=trained["runtime"],
              let torch=runtime["torch"]?.stringValue,let epochs=checkpoint["completedEpoch"]?.integerValue,
              case .array(let trainingIDs)=provenance["trainingRowIDs"],!trainingIDs.isEmpty,
              let device=manifest.runtime["device"].flatMap(ContractDevice.init(rawValue:)),
              let summaryBytes=contents["summary.json"],let raw=contents["predictions.jsonl"],let tableBytes=contents["prediction-table.json"],
              let tableFile=manifest.artifacts.first(where:{$0.path=="prediction-table.json"}) else {throw failure("Evaluation requires its captured objective, model and complete table evidence.")}
        let mapping=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(selected.targetMapping))
        let identity:JSONValue = .object(["schemaVersion":1,"recipe":selected.configuration,"targetMapping":mapping,"scalarization":.string(scalarization.rawValue)])
        guard contract["objective"]==identity,settings["objectiveRecipe"]==selected.configuration,settings["targetMapping"]==mapping,settings["objectiveScalarization"] == .string(scalarization.rawValue),
              provenance["objectiveRecipe"]==selected.configuration,provenance["objectiveMapping"]==mapping,
              provenance["objective"] == .object(["recipe":selected.configuration,"targetMapping":mapping,"scalarization":.string(scalarization.rawValue),"torchVersion":.string(torch)]),
              manifest.testProvenance["objectiveRecipe"]==selected.configuration,manifest.testProvenance["objectiveMapping"]==mapping else {throw failure("Frozen objective recipe, target roles or scalarization differ.")}
        let configuration=try LossConfiguration.decodeRecipe(selected.configuration)
        let recipe=try LossRecipe(configuration:configuration,source:selected.source,targetMapping:selected.targetMapping)
        let output=try WorkflowMappingRequirements.outputContract(capture.plan.modelGraph,registry:registry)
        let object=try JSONDecoder().decode([String:JSONValue].self,from:summaryBytes)
        guard Set(object.keys)==["schemaVersion","evaluatedCount","device","batchSize","objective","preview"] else {throw failure("Objective evaluation summary has unknown or missing fields.")}
        let summary=try JSONDecoder().decode(WorkflowEvaluationSummary.self,from:summaryBytes)
        guard summary.schemaVersion==2,(1...100000).contains(summary.evaluatedCount),summary.batchSize==request.settings.batchSize,
              summary.device==device.rawValue,request.settings.device=="auto" || request.settings.device==summary.device,
              let objective=summary.objective else {throw failure("Objective evaluation count, device or settings differ.")}
        let context=try WorkflowObjectiveEvaluationContext(recipe:recipe,runID:manifest.evaluationID,outputSampleShape:output.layout.sample.dimensions,trainingCount:trainingIDs.count,evaluationCount:summary.evaluatedCount,expectedEpochs:epochs,scalarization:scalarization,device:device,policy:.init(activationValues:false,predictionRows:request.retainComponentOutputs != false))
        let rows=try decodeRows(raw,context:context)
        guard summary.objectivePreview==Array(rows.prefix(request.retainComponentOutputs == false ? 0:request.settings.previewLimit)) else {throw failure("Objective preview differs from full predictions or capture policy.")}
        let evaluation=try verifyObjective(objective,rows:rows,context:context)
        try verifyTable(tableBytes,rows:rows)
        let table=try WorkflowPredictionTableEvidence.verify(tableBytes,file:.init(path:"artifacts/"+tableFile.path,bytes:tableFile.bytes,sha256:tableFile.sha256),partition:request.settings.partitionRole,observationIDs:rows.map(\.observationID),sourceDigest:try request.requestSHA256())
        return .init(summary:summary,rows:rows,evaluation:evaluation,table:table)
    }
    static func decodeRows(_ raw:Data,context:WorkflowObjectiveEvaluationContext)throws->[WorkflowEvaluationObjectivePrediction] {
        guard raw.count<=100_000_000,raw.last==10 else {throw failure("Objective prediction JSONL is incomplete or exceeds bounds.")}
        let lines=raw.split(separator:10,maxSplits:context.evaluationCount+1,omittingEmptySubsequences:false)
        guard lines.count==context.evaluationCount+1,lines.last?.isEmpty==true else {throw failure("Objective prediction count differs from the complete population.")}
        var rows:[WorkflowEvaluationObjectivePrediction]=[],ids=Set<String>()
        for line in lines.dropLast() {
            try Task.checkCancellation()
            guard !line.isEmpty,line.count<=1_000_000 else {throw failure("Objective prediction row exceeds bounds.")}
            let bytes=Data(line),value=try JSONDecoder().decode([String:JSONValue].self,from:bytes)
            guard Set(value.keys)==["observationID","output","targets"] else {throw failure("Objective row has unknown or missing fields.")}
            let row=try JSONDecoder().decode(WorkflowEvaluationObjectivePrediction.self,from:bytes)
            guard !row.observationID.isEmpty,row.observationID.utf8.count<=200,ids.insert(row.observationID).inserted,Set(row.targets.keys)==Set(context.recipe.targetMapping.roles.map(\.role)) else {throw failure("Objective observation or target roles differ.")}
            _ = try ObjectiveResultTensor(json:row.output,shape:context.outputSampleShape,dtype:.float32)
            for role in context.recipe.targetMapping.roles {_ = try ObjectiveResultTensor(json:row.targets[role.role]!,shape:role.sampleShape,dtype:role.dtype)}
            rows.append(row)
        }
        return rows
    }
    static func verifyObjective(_ value:JSONValue,rows:[WorkflowEvaluationObjectivePrediction],context:WorkflowObjectiveEvaluationContext)throws->ObjectiveEvaluation {
        guard case .object(let raw)=value else {throw failure("Objective summary is missing.")}
        var keys:Set<String>=["componentID","reduction","scalarization","rawShape","observations","activeCount","ignoredCount","maskedCount","effectiveScalar","aggregationNumerator","aggregationDenominator","aggregationConvention","device"]
        if raw["classification"] != nil {keys.insert("classification")}
        if context.recipe.configuration.componentID=="loss_ctc" {keys.formUnion(["alignmentWork","infeasibleCount","meanConvention"])}
        guard Set(raw.keys)==keys else {throw failure("Objective summary has unsupported fields.")}
        if case .object(let classification)=raw["classification"] {
            guard Set(classification.keys)==["rule","classAxis","ignoreIndex","eligibleCount","correctCount","accuracy"] else {throw failure("Classification summary fields differ.")}
        }
        let evaluation=try WorkflowObjectiveArtifacts.evaluation(value,context:context)
        // Independent full-population check for the regression objectives. Other losses
        // retain their shared typed reducer contract and portable worker replay check.
        if ["loss_mse","loss_l1"].contains(context.recipe.configuration.componentID) {
            guard rows.count==context.evaluationCount,let role=context.recipe.targetMapping.roles.first(where:{$0.role=="target"}) else {throw failure("Regression target mapping is missing.")}
            var sum=0.0,correction=0.0,elements=0
            for row in rows {
                try Task.checkCancellation()
                let output=try ObjectiveResultTensor(json:row.output,shape:context.outputSampleShape,dtype:.float32)
                let target=try ObjectiveResultTensor(json:row.targets["target"] ?? .null,shape:role.sampleShape,dtype:role.dtype)
                let rank=max(output.shape.count,target.shape.count),a=Array(repeating:1,count:rank-output.shape.count)+output.shape,b=Array(repeating:1,count:rank-target.shape.count)+target.shape
                guard zip(a,b).allSatisfy({$0==$1 || $0==1 || $1==1}) else {throw failure("Regression output and target shapes do not broadcast.")}
                let shape=zip(a,b).map {max($0,$1)},count=shape.reduce(1,*)
                guard count<=100_000_000,elements<=100_000_000-count else {throw failure("Regression comparison exceeds bounds.")}
                for index in 0..<count {
                    if index%256==0 {try Task.checkCancellation()}
                    var position=index,ai=0,bi=0,asize=1,bsize=1
                    for dimension in shape.indices.reversed() {let coordinate=position%shape[dimension];position/=shape[dimension];ai+=(a[dimension]==1 ? 0:coordinate)*asize;bi+=(b[dimension]==1 ? 0:coordinate)*bsize;asize*=a[dimension];bsize*=b[dimension]}
                    let delta=output.values[ai]-target.values[bi],loss=context.recipe.configuration.componentID=="loss_mse" ? delta*delta:abs(delta)
                    let adjusted=loss-correction,next=sum+adjusted;correction=(next-sum)-adjusted;sum=next
                }
                elements+=count
            }
            let denominator=evaluation.aggregationConvention=="sum" ? 1.0:Double(elements)
            func near(_ x:Double,_ y:Double)->Bool {x.isFinite && y.isFinite && abs(x-y)<=max(1e-6,max(abs(x),abs(y))*1e-5)}
            guard near(sum,evaluation.aggregationNumerator),near(denominator,evaluation.aggregationDenominator),near(sum/denominator,evaluation.effectiveScalar) else {throw failure("Regression metrics do not reproduce from full predictions.")}
        }
        return evaluation
    }
    static func verifyTable(_ bytes:Data,rows:[WorkflowEvaluationObjectivePrediction])throws {
        let table=try JSONDecoder().decode([String:JSONValue].self,from:bytes)
        let columns:[JSONValue]=["observation_id","partition","target","output"].map {.object(["name":.string($0),"dtype":.string("string"),"nullable":false])}
        guard table["columns"] == .array(columns),case .array(let values)=table["rows"],values.count==rows.count else {throw failure("Objective table schema differs.")}
        for (index,value) in values.enumerated() {
            if index%256==0 {try Task.checkCancellation()}
            guard case .array(let cells)=value,cells.count==4,cells[0] == .string(rows[index].observationID),let target=cells[2].stringValue,let output=cells[3].stringValue,
                  try JSONDecoder().decode(JSONValue.self,from:Data(target.utf8)) == .object(rows[index].targets),try JSONDecoder().decode(JSONValue.self,from:Data(output.utf8)) == rows[index].output else {throw failure("Objective table differs from verified full predictions.")}
        }
    }
    private static func failure(_ message:String)->WorkflowArchiveError {.init(message)}
}
