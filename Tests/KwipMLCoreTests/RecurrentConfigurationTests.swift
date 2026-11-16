import XCTest
@testable import KwipMLCore

final class RecurrentConfigurationTests: XCTestCase {
    func testEveryDefaultUsesExplicitResetAndNativeParameterOwnership() throws {
        for id in RecurrentConfiguration.componentIDs {
            let c = try RecurrentConfiguration(componentID:id,values:[:])
            XCTAssertEqual(c.configuredInputs,["in"])
            XCTAssertEqual(c.values["state_mode"],.string("zero"))
            XCTAssertEqual(c.parameterShapes.first?.name,id.hasSuffix("_cell") ? "weight_ih":"weight_ih_l0")
            let gates = id.hasPrefix("lstm") ? 4 : id.hasPrefix("gru") ? 3 : 1
            XCTAssertEqual(c.parameterCount,gates * 12)
        }
    }
    func testProjectedBidirectionalStateGeometryAndNativeOrder() throws {
        let c = try RecurrentConfiguration(componentID:"lstm",values:["input_size":3,"hidden_size":4,"proj_size":2,"num_layers":2,"bidirectional":true,"state_mode":.string("supplied")])
        XCTAssertEqual(c.configuredInputs,["in","hidden","cell"])
        let r=try c.resolved(layouts:[.init(sample:[3,3]),.init(sample:[4,2]),.init(sample:[4,4])])
        XCTAssertEqual(r.outputs["out"]?.sample.dimensions,[3,4])
        XCTAssertEqual(r.outputs["hidden"]?.sample.dimensions,[4,2])
        XCTAssertEqual(r.outputs["cell"]?.sample.dimensions,[4,4])
        XCTAssertEqual(c.parameterCount,512)
        XCTAssertEqual(c.parameterShapes.map(\.name).prefix(6),["weight_ih_l0","weight_hh_l0","bias_ih_l0","bias_hh_l0","weight_hr_l0","weight_ih_l0_reverse"])
        XCTAssertTrue(c.supports(device:"cpu"));XCTAssertFalse(c.supports(device:"mps"))
    }
    func testCellHiddenIsAliasAndNamedStateNeverConsumesObservationAxis() throws {
        let c=try RecurrentConfiguration(componentID:"lstm_cell",values:["state_mode":.string("supplied")])
        let r=try c.resolved(layouts:[.init(sample:[2]),.init(sample:[2]),.init(sample:[2])])
        XCTAssertEqual(r.outputs["out"],r.outputs["hidden"])
        XCTAssertEqual(r.outputElementsPerObservation,4)
        XCTAssertThrowsError(try c.resolved(layouts:[.init(sample:[2]),.init(sample:[1,2]),.init(sample:[2])]))
    }
    func testInvalidControlsAndAllocationRefuseBeforeExecution() throws {
        for config in ([["input_size":true],["hidden_size":0],["bias":1],["state_mode":.string("remember")],["num_layers":1,"dropout":0.2],["proj_size":2],["unknown":1]] as [[String:JSONValue]]) {
            XCTAssertThrowsError(try RecurrentConfiguration(componentID:"lstm",values:config))
        }
        let c=try RecurrentConfiguration(componentID:"rnn",values:["input_size":4096,"hidden_size":512])
        XCTAssertThrowsError(try c.resolved(layouts:[.init(sample:[4096,4096])]))
    }
}
