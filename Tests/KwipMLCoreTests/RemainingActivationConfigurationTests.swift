import XCTest
@testable import KwipMLCore

final class RemainingActivationConfigurationTests:XCTestCase {
    func testDefaultsCanonicalizeTheActualDeclaredFields()throws {
        let p=try RemainingActivationConfiguration(componentID:"prelu",values:[:])
        XCTAssertEqual(p.values,["num_parameters":1,"init":.number(0.25)])
        XCTAssertEqual(p.numParameters,1);XCTAssertEqual(p.initialSlope,0.25)
        let l=try RemainingActivationConfiguration(componentID:"log_softmax",values:[:])
        XCTAssertEqual(l.values,["dim":-1]);XCTAssertEqual(l.dim,-1)
        let d=try RemainingActivationConfiguration(componentID:"dropout",values:[:])
        XCTAssertEqual(d.values,["p":.number(0.5)]);XCTAssertEqual(d.probability,0.5)
    }
    func testPReLUCountsChannelsAtFirstSampleDirectionRatherThanColumns()throws {
        let shape=try BatchedTensorLayout(sample:[2,3])
        let p=try RemainingActivationConfiguration(componentID:"prelu",values:["num_parameters":2,"init":-0.5])
        XCTAssertEqual(try p.resolved(layout:shape),shape)
        XCTAssertEqual(try p.parameterCount(for:shape),2)
        XCTAssertEqual(p.initialSlope,-0.5)
        XCTAssertThrowsError(try RemainingActivationConfiguration(componentID:"prelu",values:["num_parameters":3]).resolved(layout:shape))
        XCTAssertEqual(try RemainingActivationConfiguration(componentID:"prelu",values:[:]).parameterCount(for:shape),1)
        let maximum=try BatchedTensorLayout(sample:[4096,1,1,1,1,1,1])
        XCTAssertEqual(try RemainingActivationConfiguration(componentID:"prelu",values:["num_parameters":4096,"init":0]).resolved(layout:maximum),maximum)
    }
    func testLogSoftmaxSignedAxesPreserveShapeButNeverNormalizeObservations()throws {
        let shape=try BatchedTensorLayout(sample:[2,3,4])
        for axis in [1,2,3,-1,-2,-3] {
            let c=try RemainingActivationConfiguration(componentID:"log_softmax",values:["dim":.integer(Int64(axis))])
            XCTAssertEqual(try c.resolved(layout:shape),shape)
            XCTAssertEqual(try c.parameterCount(for:shape),0)
        }
        for axis in [0,-4,4,-5] {
            XCTAssertThrowsError(try RemainingActivationConfiguration(componentID:"log_softmax",values:["dim":.integer(Int64(axis))]).resolved(layout:shape))
        }
    }
    func testDropoutEndpointsRemainConfigurableWithoutNodeModeOrSeedOverrides()throws {
        let shape=try BatchedTensorLayout(sample:[2,3])
        for p in [0.0,0.25,0.5,1.0] {
            let c=try RemainingActivationConfiguration(componentID:"dropout",values:["p":.number(p)])
            XCTAssertEqual(c.probability,p);XCTAssertEqual(try c.resolved(layout:shape),shape)
            XCTAssertEqual(try c.parameterCount(for:shape),0)
        }
        for config:[String:JSONValue] in [["p":-0.01],["p":1.01],["p":true],["p":.null],["p":.number(.infinity)],["seed":7],["training":true],["inplace":true]] {
            XCTAssertThrowsError(try RemainingActivationConfiguration(componentID:"dropout",values:config))
        }
    }
    func testStrictUnknownTypesFractionalCountsAndNonfiniteSlopesRefuse() {
        for config:[String:JSONValue] in [["num_parameters":true],["num_parameters":.number(2.5)],["num_parameters":0],["num_parameters":4097],["num_parameters":.null],["init":true],["init":.null],["init":.number(.nan)],["init":.number(Double(Float.greatestFiniteMagnitude)*2)],["channels":2]] {
            XCTAssertThrowsError(try RemainingActivationConfiguration(componentID:"prelu",values:config))
        }
        for config:[String:JSONValue] in [["dim":true],["dim":.number(1.5)],["dim":.null],["dim":-9],["dim":8],["dtype":.string("float64")]] {
            XCTAssertThrowsError(try RemainingActivationConfiguration(componentID:"log_softmax",values:config))
        }
        XCTAssertThrowsError(try RemainingActivationConfiguration(componentID:"relu",values:[:]))
    }
    func testAllThreeRequireFloat32ContractsWithoutImplicitConversion()throws {
        let layout=try BatchedTensorLayout(sample:[2,3])
        for id in RemainingActivationConfiguration.componentIDs {
            let c=try RemainingActivationConfiguration(componentID:id,values:[:])
            XCTAssertEqual(try c.resolved(input:.init(layout:layout,dtype:.float32)),.init(layout:layout,dtype:.float32))
            for dtype:ContractDType in [.float64,.int64,.int32,.bool,.string] {
                XCTAssertThrowsError(try c.resolved(input:.init(layout:layout,dtype:dtype)))
            }
        }
    }
}
