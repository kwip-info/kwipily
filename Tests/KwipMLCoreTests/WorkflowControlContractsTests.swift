import XCTest
@testable import KwipMLCore

final class WorkflowControlContractsTests:XCTestCase {
    private let address=WorkflowNodeAddress(domain:.control,localID:"setup")
    func testVariableMappingPreservesOrderAndRejectsMalformedOrOverboundColumns()throws {
        for count in [1,3,128] {
            let names=(0..<count).reversed().map {"column \($0)"}
            let value=try WorkflowControlContracts.trainingMapping(["featureColumns":.array(names.map(JSONValue.string)),"targetColumn":.string("label")],address:address)
            XCTAssertEqual(value.0,names)
        }
        let invalid:[[JSONValue]] = [[],[.string("a"),true],[.string("a"),.string("a")],(0..<129).map {JSONValue.string("c\($0)")}]
        for columns in invalid {
            XCTAssertThrowsError(try WorkflowControlContracts.trainingMapping(["featureColumns":.array(columns),"targetColumn":.string("label")],address:address))
        }
    }
    func testRunContractDefaultsAndStrictBoundaryTypes()throws {
        XCTAssertEqual(try WorkflowControlContracts.runSettings([:],address:address),ExecutionSettings())
        let valid:[String:JSONValue] = ["epochs":10000,"batchSize":100000,"seed":4294967295,"learningRate":1,"device":.string("mps")]
        let result=try WorkflowControlContracts.runSettings(valid,address:address)
        XCTAssertEqual(result.epochs,10000);XCTAssertEqual(result.seed,4294967295);XCTAssertEqual(result.learningRate,1)
        let invalid:[(String,JSONValue)] = [("epochs",0),("epochs",10001),("epochs",.number(2)),("batchSize",100001),("seed",-1),("seed",4294967296),("learningRate",0),("learningRate",.number(.infinity)),("device",.string("cuda")),("unknown",1)]
        for (key,value) in invalid {
            XCTAssertThrowsError(try WorkflowControlContracts.runSettings([key:value],address:address),"\(key): \(value)")
        }
        XCTAssertEqual(Set(WorkflowControlContracts.runFields.map(\.key)),["device","epochs","batchSize","seed","learningRate"])
    }
    func testMappingContractRetainsExactNamesAndRequiresSeparateTarget()throws {
        let config:[String:JSONValue] = ["featureColumns":.array([.string(" x, "),.string("second")]),"targetColumn":.string(" target ")]
        let value=try WorkflowControlContracts.trainingMapping(config,address:address)
        XCTAssertEqual(value.0,[" x, ","second"]);XCTAssertEqual(value.1," target ")
        XCTAssertThrowsError(try WorkflowControlContracts.trainingMapping([:],address:address))
        var invalid=config;invalid["targetColumn"] = .string("second")
        XCTAssertThrowsError(try WorkflowControlContracts.trainingMapping(invalid,address:address))
        invalid=config;invalid["featureColumns"] = .array([.string(" "),.string("second")])
        XCTAssertThrowsError(try WorkflowControlContracts.trainingMapping(invalid,address:address))
    }
    func testGradientCaptureUsesStrictExistingConfigurationContract()throws {
        let value:JSONValue = .object(["schemaVersion":1,"everyEpochs":1,"maxBatchesPerEpoch":8,"maxCaptures":1024])
        XCTAssertEqual(try WorkflowControlContracts.runSettings(["gradientCapture":value],address:address).gradientCapture,GradientCaptureConfiguration(everyEpochs:1,maxBatchesPerEpoch:8,maxCaptures:1024))
        XCTAssertNil(try WorkflowControlContracts.runSettings(["gradientCapture":.null],address:address).gradientCapture)
        XCTAssertThrowsError(try WorkflowControlContracts.runSettings(["gradientCapture":.object(["schemaVersion":1,"everyEpochs":1,"maxBatchesPerEpoch":9,"maxCaptures":1024])],address:address))
    }
}
