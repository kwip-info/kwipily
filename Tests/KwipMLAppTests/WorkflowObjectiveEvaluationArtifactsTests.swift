import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowObjectiveEvaluationArtifactsTests:XCTestCase {
    private func context(width:Int=2,reduction:String="mean",broadcast:Bool=false)throws->WorkflowObjectiveEvaluationContext {
        let configuration=try LossConfiguration(componentID:"loss_mse",values:["reduction":.string(reduction),"target_broadcast":.bool(broadcast)])
        let mapping=LossTargetMapping(componentID:"loss_mse",inputColumns:["x"],roles:[.init(role:"target",columns:broadcast ? ["y"]:(0..<width).map {"y\($0)"},sampleShape:broadcast ? [1]:[width],dtype:.float32)])
        let recipe=try LossRecipe(configuration:configuration,source:.init(domain:.control,localID:"train"),targetMapping:mapping)
        return try .init(recipe:recipe,runID:"evaluation",outputSampleShape:[width],trainingCount:4,evaluationCount:2,expectedEpochs:1,scalarization:.mean,device:.cpu,policy:.init(activationValues:false,predictionRows:true))
    }
    private func rows(width:Int=2)->[WorkflowEvaluationObjectivePrediction] {
        [.init(observationID:"a",output:.array(Array(repeating:.integer(2),count:width)),targets:["target":.array(Array(repeating:.integer(1),count:width))]),
         .init(observationID:"b",output:.array(Array(repeating:.integer(4),count:width)),targets:["target":.array(Array(repeating:.integer(2),count:width))])]
    }
    private func jsonl(_ rows:[WorkflowEvaluationObjectivePrediction])throws->Data {
        var result=Data();for row in rows {result.append(try JSONEncoder().encode(row));result.append(10)};return result
    }
    private func objective(width:Int=2)->JSONValue {.object(["componentID":.string("loss_mse"),"reduction":.string("mean"),"scalarization":.null,"rawShape":.array([]),"observations":2,"activeCount":2,"ignoredCount":0,"maskedCount":0,"effectiveScalar":2.5,"aggregationNumerator":.number(Double(5*width)),"aggregationDenominator":.number(Double(2*width)),"aggregationConvention":.string("nativeMean"),"device":.string("cpu")])}
    func testScalarAndVectorRegressionRemainTypedAndNeverAcquireClassAccuracy()throws {
        for width in [1,2] {
            let c=try context(width:width),expected=rows(width:width)
            let decoded=try WorkflowObjectiveEvaluationArtifacts.decodeRows(jsonl(expected),context:c)
            XCTAssertEqual(decoded,expected)
            let evaluation=try WorkflowObjectiveEvaluationArtifacts.verifyObjective(objective(width:width),rows:decoded,context:c)
            XCTAssertEqual(evaluation.effectiveScalar,2.5);XCTAssertNil(evaluation.accuracy)
            let summary:JSONValue = .object(["schemaVersion":2,"evaluatedCount":2,"device":.string("cpu"),"batchSize":1,"objective":objective(width:width),"preview":.array([])])
            let parsed=try JSONDecoder().decode(WorkflowEvaluationSummary.self,from:JSONEncoder().encode(summary))
            XCTAssertNil(parsed.accuracy);XCTAssertNil(parsed.lossMean);XCTAssertTrue(parsed.confusionMatrix.isEmpty)
        }
    }
    func testRegressionMetricForgeryAndClassificationClaimsAreRefused()throws {
        let c=try context(),r=rows()
        guard case .object(let original)=objective() else {return XCTFail()}
        for field in ["aggregationNumerator","aggregationDenominator","effectiveScalar"] {
            var changed=original;changed[field]=99
            XCTAssertThrowsError(try WorkflowObjectiveEvaluationArtifacts.verifyObjective(.object(changed),rows:r,context:c))
        }
        var classification=original;classification["classification"] = .object(["rule":.string("argmax"),"classAxis":-1,"ignoreIndex":-100,"eligibleCount":2,"correctCount":2,"accuracy":1])
        XCTAssertThrowsError(try WorkflowObjectiveEvaluationArtifacts.verifyObjective(.object(classification),rows:r,context:c))
        var unknown=original;unknown["accuracy"]=1
        XCTAssertThrowsError(try WorkflowObjectiveEvaluationArtifacts.verifyObjective(.object(unknown),rows:r,context:c))
        let changed=[WorkflowEvaluationObjectivePrediction(observationID:"a",output:.array([9,9]),targets:["target":.array([1,1])]),r[1]]
        XCTAssertThrowsError(try WorkflowObjectiveEvaluationArtifacts.verifyObjective(objective(),rows:changed,context:c))
    }
    func testMalformedRolesGeometryDuplicateAndIncompletePopulationAreRefused()throws {
        let c=try context(),r=rows()
        for invalid in [Array(r.prefix(1)),[r[0],r[0]],[.init(observationID:"a",output:.array([2]),targets:["target":.array([1,1])]),r[1]],[.init(observationID:"a",output:.array([2,2]),targets:["other":.array([1,1])]),r[1]]] {
            XCTAssertThrowsError(try WorkflowObjectiveEvaluationArtifacts.decodeRows(jsonl(invalid),context:c))
        }
        var missingNewline=try jsonl(r);missingNewline.removeLast()
        XCTAssertThrowsError(try WorkflowObjectiveEvaluationArtifacts.decodeRows(missingNewline,context:c))
        let old=Data("{\"observationID\":\"a\",\"logits\":[2,2],\"predicted\":0,\"target\":0}\n".utf8)
        XCTAssertThrowsError(try WorkflowObjectiveEvaluationArtifacts.decodeRows(old,context:c))
    }
    func testStandardTablePreservesBothOutputsAndTargetsAndRejectsTampering()throws {
        let r=rows(),columns:[JSONValue]=["observation_id","partition","target","output"].map {.object(["name":.string($0),"dtype":.string("string"),"nullable":false])}
        let values: [JSONValue] = r.map {.array([.string($0.observationID),.string("test"),.string($0.observationID=="a" ? "{\"target\":[1,1]}":"{\"target\":[2,2]}"),.string($0.observationID=="a" ? "[2,2]":"[4,4]")])}
        let table:JSONValue = .object(["columns":.array(columns),"rows":.array(values)])
        XCTAssertNoThrow(try WorkflowObjectiveEvaluationArtifacts.verifyTable(JSONEncoder().encode(table),rows:r))
        let changed=[WorkflowEvaluationObjectivePrediction(observationID:"a",output:.array([2,3]),targets:["target":.array([1,1])]),r[1]]
        XCTAssertThrowsError(try WorkflowObjectiveEvaluationArtifacts.verifyTable(JSONEncoder().encode(table),rows:changed))
    }
    func testBroadcastUsesExplicitTargetGeometryRatherThanFlatteningOrArgmax()throws {
        let c=try context(broadcast:true)
        let r:[WorkflowEvaluationObjectivePrediction]=[.init(observationID:"a",output:.array([2,2]),targets:["target":.array([1])]),.init(observationID:"b",output:.array([4,4]),targets:["target":.array([2])])]
        XCTAssertNoThrow(try WorkflowObjectiveEvaluationArtifacts.decodeRows(jsonl(r),context:c))
        XCTAssertEqual(try WorkflowObjectiveEvaluationArtifacts.verifyObjective(objective(),rows:r,context:c).effectiveScalar,2.5)
    }
}
