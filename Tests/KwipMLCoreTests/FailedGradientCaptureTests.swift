import XCTest
@testable import KwipMLCore

final class FailedGradientCaptureTests: XCTestCase {
    let runID = "11111111-1111-4111-8111-111111111111"
    func graph() -> GraphDocument {
        GraphDocument(nodes: [GraphNode(id:"a",type:"linear",config:["in_features":2,"out_features":1]),GraphNode(id:"b",type:"bilinear",config:["in1_features":1,"in2_features":1,"out_features":1,"bias":true])])
    }
    func bytes(_ value: Any) throws -> Data { try JSONSerialization.data(withJSONObject:value,options:.sortedKeys) }
    func context() throws -> FailedGradientCaptureContext {
        let parameters: [[String:Any]] = [("a","weight",[1,2]),("a","bias",[1]),("b","weight",[1,1,1]),("b","bias",[1])].map { node,kind,shape in
            ["name":"layers.\(node).\(kind)","nodeID":node,"shape":shape,"elementCount":shape.reduce(1,*)]
        }
        return try FailedGradientCaptureCodec.decodeContext(bytes([
            "workerRunID":runID,"device":"cpu","initialEpoch":0,"requestedFinalEpoch":2,"requestedBatchSize":32,"trainingRows":48,
            "configuration":["schemaVersion":1,"everyEpochs":1,"maxBatchesPerEpoch":1,"maxCaptures":4],
            "inventory":["parameterCount":4,"elementCount":5,"parameters":parameters]
        ]),graph:graph())
    }
    func event() -> [String:Any] {
        ["event":"gradientInterrupted","schemaVersion":1,"id":"kwip.failed-gradient-observations.v1","runID":runID,"epoch":1,"stage":"trainingForward","lastBackwardBatchIndex":0,"totalBatches":2,"totalTrainingRows":48,"scheduledBatchCount":1,"observedBatchCount":1,"observedTrainingRows":32,"captureBudgetRemaining":3,
         "batches":[["batchIndex":0,"trainingRows":32,"presentParameterCount":4,"presentElementCount":5,"missingParameterCount":0,"missingElementCount":0,"finiteElementCount":2,"nonfiniteElementCount":3,"zeroElementCount":1,"l2Norm":NSNull(),"maxAbs":NSNull()]]]
    }
    func testConfigurableActivationsAddNoInventedParameters() throws {
        let context = try context()
        for id in ConfigurableActivationRuntimeCapabilities.operations {
            var changed = graph()
            changed.nodes.append(GraphNode(id: "response", type: id, config: [:]))
            XCTAssertNoThrow(try FailedGradientCaptureCodec.validate(context, graph: changed), id)
        }
        var unknown = graph()
        unknown.nodes.append(GraphNode(id: "response", type: "unreviewed-activation", config: [:]))
        XCTAssertThrowsError(try FailedGradientCaptureCodec.validate(context, graph: unknown))
    }
    func testAdaptivePoolingAddsNoInventedParameters() throws {
        let context = try context()
        for id in AdaptivePoolingConfiguration.componentIDs {
            var changed = graph()
            changed.nodes.append(GraphNode(id: "summary", type: id, config: [:]))
            XCTAssertNoThrow(try FailedGradientCaptureCodec.validate(context, graph: changed), id)
        }
    }
    func testPartialNonfiniteBilinearInventoryAndTrueCoverage() throws {
        let context=try context()
        let result=try FailedGradientCaptureCodec.decodeInterrupted(bytes(event()),context:context,previous:[])
        XCTAssertEqual(result.batches.first?.nonfiniteElementCount,3)
        XCTAssertEqual(result.lastBackwardBatchIndex,0)
        XCTAssertNil(result.batches.first?.l2Norm)
    }
    func testIdentityCadenceInventoryAndPartialBoundaryRefuseTampering() throws {
        let context=try context()
        for (key,value) in [("runID",UUID().uuidString as Any),("epoch",2),("lastBackwardBatchIndex",NSNull()),("captureBudgetRemaining",4),("observedTrainingRows",31),("stage","validationLoss"),("unknown",true)] {
            var changed=event();changed[key]=value
            XCTAssertThrowsError(try FailedGradientCaptureCodec.decodeInterrupted(bytes(changed),context:context,previous:[]),key)
        }
        var changed=event();var batches=changed["batches"] as! [[String:Any]];batches[0]["trainingRows"]=16;changed["batches"]=batches
        XCTAssertThrowsError(try FailedGradientCaptureCodec.decodeInterrupted(bytes(changed),context:context,previous:[]))
        var wrong=graph();wrong.nodes[1].config["in1_features"]=2
        XCTAssertThrowsError(try FailedGradientCaptureCodec.validate(context,graph:wrong))
    }
    func testCompletedPrefixAndInterruptedEpochKeepRequestedCoverageDistinct()throws {
        let context=try context()
        var complete=event();complete.removeValue(forKey:"id");complete.removeValue(forKey:"stage");complete.removeValue(forKey:"lastBackwardBatchIndex");complete["event"]="gradientSummary";complete["status"]="captured"
        let previous=try FailedGradientCaptureCodec.decodeCompleted(bytes(complete),context:context,previous:[])
        var partial=event();partial["epoch"]=2;partial["captureBudgetRemaining"]=2
        let current=try FailedGradientCaptureCodec.decodeInterrupted(bytes(partial),context:context,previous:[previous])
        XCTAssertEqual(current.epoch,2);XCTAssertEqual(current.observedBatchCount,1)
        XCTAssertEqual(context.requestedFinalEpoch,2)
        partial["epoch"]=3
        XCTAssertThrowsError(try FailedGradientCaptureCodec.decodeInterrupted(bytes(partial),context:context,previous:[previous]))
    }
    func testBeforeBackwardAndBudgetExhaustionDoNotInventObservations()throws {
        var context=try context()
        var empty=event();empty["lastBackwardBatchIndex"]=NSNull();empty["observedBatchCount"]=0;empty["observedTrainingRows"]=0;empty["captureBudgetRemaining"]=4;empty["batches"]=[]
        let first=try FailedGradientCaptureCodec.decodeInterrupted(bytes(empty),context:context,previous:[])
        XCTAssertTrue(first.batches.isEmpty);XCTAssertNil(first.lastBackwardBatchIndex)
        context.configuration.maxCaptures=1
        var complete=event();complete.removeValue(forKey:"id");complete.removeValue(forKey:"stage");complete.removeValue(forKey:"lastBackwardBatchIndex");complete["event"]="gradientSummary";complete["status"]="captured";complete["captureBudgetRemaining"]=0
        let previous=try FailedGradientCaptureCodec.decodeCompleted(bytes(complete),context:context,previous:[])
        empty["epoch"]=2;empty["lastBackwardBatchIndex"]=0;empty["captureBudgetRemaining"]=0
        let exhausted=try FailedGradientCaptureCodec.decodeInterrupted(bytes(empty),context:context,previous:[previous])
        XCTAssertEqual(exhausted.scheduledBatchCount,1);XCTAssertEqual(exhausted.observedBatchCount,0)
    }

}
