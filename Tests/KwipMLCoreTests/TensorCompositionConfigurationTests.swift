import XCTest
@testable import KwipMLCore

final class TensorCompositionConfigurationTests:XCTestCase {
    func testLegacyDefaultsNormalizeWithoutGuessingNewAxes()throws {
        XCTAssertEqual(try TensorCompositionConfiguration(componentID:"concat",values:[:]).values,["dim":-1])
        XCTAssertEqual(try TensorCompositionConfiguration(componentID:"add",values:[:]).values,["alpha":1,"sample_broadcast":false])
        XCTAssertEqual(try TensorCompositionConfiguration(componentID:"stack",values:[:]).values,["dim":-1])
        XCTAssertEqual(try TensorCompositionConfiguration(componentID:"permute",values:[:]).values,["dims":.array([0,2,1])])
    }
    func testLegacyCompositionSemanticIdentityOmitsDefaultsButRetainsBehaviorChanges() throws {
        let registry = try ComponentRegistry.builtIn()
        for (kind, defaults, changed): (String, [String: JSONValue], [String: JSONValue]) in [
            ("concat", ["dim": -1], ["dim": 1]),
            ("add", ["alpha": 1, "sample_broadcast": false], ["alpha": 0.5, "sample_broadcast": true])
        ] {
            let legacy = GraphDocument(schemaVersion: 1, nodes: [GraphNode(id: "merge", type: kind)])
            let explicit = GraphDocument(schemaVersion: 2, nodes: [GraphNode(id: "merge", type: kind, config: defaults)])
            let altered = GraphDocument(schemaVersion: 2, nodes: [GraphNode(id: "merge", type: kind, config: changed)])
            let legacyBytes = try GraphSemantics.canonicalData(legacy, registry: registry)
            let object = try XCTUnwrap(JSONSerialization.jsonObject(with: legacyBytes) as? [String: Any])
            let nodes = try XCTUnwrap(object["nodes"] as? [[String: Any]])
            XCTAssertEqual(try XCTUnwrap(nodes.first?["config"] as? [String: Any]).count, 0)
            XCTAssertEqual(legacyBytes, try GraphSemantics.canonicalData(explicit, registry: registry))
            XCTAssertNotEqual(legacyBytes, try GraphSemantics.canonicalData(altered, registry: registry))
        }
    }
    func testConcatenateNondefaultExistingDirectionAndBatchRefusal()throws {
        let layouts=[try BatchedTensorLayout(sample:[2,3]),try BatchedTensorLayout(sample:[1,3])]
        for axis in [1,-2] {
            let c=try TensorCompositionConfiguration(componentID:"concat",values:["dim":.integer(Int64(axis))])
            XCTAssertEqual(try c.resolved(layouts:layouts).sample.dimensions,[3,3])
        }
        XCTAssertThrowsError(try TensorCompositionConfiguration(componentID:"concat",values:["dim":-3]).resolved(layouts:layouts))
        XCTAssertThrowsError(try TensorCompositionConfiguration(componentID:"concat",values:["dim":1]).resolved(layouts:[layouts[0],try .init(sample:[1,4])]))
    }
    func testAddRightAlignsOnlySampleDirectionsAndRequiresOptIn()throws {
        let c=try TensorCompositionConfiguration(componentID:"add",values:["alpha":-0.5,"sample_broadcast":true])
        XCTAssertEqual(c.alpha,-0.5);XCTAssertTrue(c.sampleBroadcast)
        XCTAssertEqual(try c.resolved(layouts:[.init(sample:[2,1]),.init(sample:[3])]).sample.dimensions,[2,3])
        XCTAssertThrowsError(try TensorCompositionConfiguration(componentID:"add",values:[:]).resolved(layouts:[.init(sample:[2,1]),.init(sample:[3])]))
        XCTAssertThrowsError(try c.resolved(layouts:[.init(sample:[2,3]),.init(sample:[4])]))
    }
    func testStackInsertsDirectionIncludingLastAxisAtMaximumInputRank()throws {
        let shape=try BatchedTensorLayout(sample:[2,3])
        XCTAssertEqual(try TensorCompositionConfiguration(componentID:"stack",values:["dim":2]).resolved(layouts:[shape,shape]).sample.dimensions,[2,2,3])
        let maximum=try BatchedTensorLayout(sample:[1,1,1,1,1,1])
        XCTAssertEqual(try TensorCompositionConfiguration(componentID:"stack",values:["dim":7]).resolved(layouts:[maximum,maximum]).sample.dimensions,[1,1,1,1,1,1,2])
        let tooMany=try BatchedTensorLayout(sample:[1,1,1,1,1,1,1])
        XCTAssertThrowsError(try TensorCompositionConfiguration(componentID:"stack",values:["dim":-1]).resolved(layouts:[tooMany,tooMany]))
        XCTAssertThrowsError(try TensorCompositionConfiguration(componentID:"stack",values:["dim":-4]).resolved(layouts:[shape,shape]))
    }
    func testPermuteNonInvolutiveOrderNegativeAliasesAndMovedBatchRefuse()throws {
        let layout=try BatchedTensorLayout(sample:[2,3,4])
        for order:[JSONValue] in [[0,2,3,1],[0,-2,-1,-3]] {
            XCTAssertEqual(try TensorCompositionConfiguration(componentID:"permute",values:["dims":.array(order)]).resolved(layouts:[layout]).sample.dimensions,[3,4,2])
        }
        for order:[JSONValue] in [[1,0,2,3],[0,1,1,3],[0,1,2],[0,1,2,4]] {
            XCTAssertThrowsError(try TensorCompositionConfiguration(componentID:"permute",values:["dims":.array(order)]).resolved(layouts:[layout]))
        }
    }
    func testInvalidFieldsNumericTypesAndNonfiniteCoefficientsRefuse()throws {
        for values:[String:JSONValue] in [["dim":true],["dim":1.5],["dim":0],["dim":8],["unknown":1]] {
            XCTAssertThrowsError(try TensorCompositionConfiguration(componentID:"concat",values:values))
        }
        for values:[String:JSONValue] in [["alpha":true],["alpha":.null],["alpha":.number(.infinity)],["alpha":1e100],["sample_broadcast":1],["unknown":false]] {
            XCTAssertThrowsError(try TensorCompositionConfiguration(componentID:"add",values:values))
        }
        XCTAssertThrowsError(try TensorCompositionConfiguration(componentID:"unknown",values:[:]))
        XCTAssertThrowsError(try TensorCompositionConfiguration(componentID:"permute",values:["dims":.array([0,true,1])]))
    }
    func testTypedContractsPreserveIntegerLegacyAndRefusePromotion()throws {
        let a=try TypedTensorContract(sample:[2,3],dtype:.int64),b=try TypedTensorContract(sample:[2,3],dtype:.float32)
        let plain=try TensorCompositionConfiguration(componentID:"add",values:[:])
        XCTAssertEqual(try plain.resolved(inputs:[a,a]).dtype,.int64)
        XCTAssertThrowsError(try plain.resolved(inputs:[a,b]))
        XCTAssertThrowsError(try TensorCompositionConfiguration(componentID:"add",values:["alpha":2]).resolved(inputs:[a,a]))
        XCTAssertEqual(try TensorCompositionConfiguration(componentID:"permute",values:["dims":.array([0,2,1])]).resolved(inputs:[a]).layout.sample.dimensions,[3,2])
    }
}
