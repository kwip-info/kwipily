import XCTest
@testable import KwipMLCore

final class LossTargetMappingTests: XCTestCase {
    private func target(_ columns:[String] = ["label"], shape:[Int] = [], dtype:ContractDType = .int64, role:String = "target") -> LossTargetRoleMapping {
        .init(role:role,columns:columns,sampleShape:shape,dtype:dtype)
    }
    func testOrdinaryMappingsPreserveOrderedColumnsAndRejectMalformedBindings() throws {
        let mapping=LossTargetMapping(componentID:"loss_mse",inputColumns:["x","y"],roles:[target(["b","a"],shape:[2],dtype:.float32)])
        XCTAssertNoThrow(try mapping.validate())
        XCTAssertEqual(try JSONDecoder().decode(LossTargetMapping.self,from:JSONEncoder().encode(mapping)),mapping)
        for roles in [[],[target(role:"answer")],[target(),target()],[target(["x"])],[target(["a","a"],shape:[2])],[target(["a"],shape:[2])],[target(["a"],shape:[0])],[target(["a"],shape:[Int.max,2])]] {
            XCTAssertThrowsError(try LossTargetMapping(componentID:"loss_mse",inputColumns:["x","y"],roles:roles).validate())
        }
        for inputs in [[],["x","x"],[" "],[String(repeating:"x",count:201)]] {
            XCTAssertThrowsError(try LossTargetMapping(componentID:"loss_mse",inputColumns:inputs,roles:[target()]).validate())
        }
        XCTAssertThrowsError(try LossTargetMapping(componentID:"unknown",inputColumns:["x"],roles:[target()]).validate())
    }
    func testCTCLengthRolesAreIntegerScalarsAndNeverShareSourceOrTargetColumns() throws {
        let roles=[target(["t0","t1"],shape:[2]),target(["source_count"],role:"input_lengths"),target(["target_count"],role:"target_lengths")]
        XCTAssertNoThrow(try LossTargetMapping(componentID:"loss_ctc",inputColumns:["x"],roles:roles).validate())
        XCTAssertThrowsError(try LossTargetMapping(componentID:"loss_ctc",inputColumns:["x"],roles:[roles[0]]).validate())
        var wrong=roles;wrong[1].dtype = .float32
        XCTAssertThrowsError(try LossTargetMapping(componentID:"loss_ctc",inputColumns:["x"],roles:wrong).validate())
        wrong=roles;wrong[1].columns=["t0"]
        XCTAssertThrowsError(try LossTargetMapping(componentID:"loss_ctc",inputColumns:["x"],roles:wrong).validate())
        wrong=roles;wrong[2].sampleShape=[1]
        XCTAssertThrowsError(try LossTargetMapping(componentID:"loss_ctc",inputColumns:["x"],roles:wrong).validate())
    }
    func testTripletHasThreeExplicitOrderedSourceGroupsAndNoDummyTarget() throws {
        let groups=[["a0","a1"],["p0","p1"],["n0","n1"]],columns=groups.flatMap {$0}
        let mapping=LossTargetMapping(componentID:"loss_triplet_margin",inputColumns:columns,roles:[],tripletSourceGroups:groups)
        XCTAssertNoThrow(try mapping.validate())
        for bad in [[],[groups[0],groups[1]],[groups[1],groups[0],groups[2]],[["a0"],groups[1],groups[2]]] {
            XCTAssertThrowsError(try LossTargetMapping(componentID:"loss_triplet_margin",inputColumns:columns,roles:[],tripletSourceGroups:bad).validate())
        }
        XCTAssertThrowsError(try LossTargetMapping(componentID:"loss_triplet_margin",inputColumns:columns,roles:[target()],tripletSourceGroups:groups).validate())
        XCTAssertThrowsError(try LossTargetMapping(componentID:"loss_mse",inputColumns:columns,roles:[target()],tripletSourceGroups:groups).validate())
    }
    func testOptionalBCEWeightsHaveExplicitShapesAndOnlyBCEAdmitsTheirRoles() throws {
        let roles=[target(["t0","t1"],shape:[2],dtype:.float32),target(["w0","w1"],shape:[2],dtype:.float32,role:"weight"),target(["p0","p1"],shape:[2],dtype:.float32,role:"pos_weight")]
        XCTAssertNoThrow(try LossTargetMapping(componentID:"loss_bce_with_logits",inputColumns:["x"],roles:roles).validate())
        XCTAssertThrowsError(try LossTargetMapping(componentID:"loss_mse",inputColumns:["x"],roles:roles).validate())
        var wrong=roles;wrong[1].dtype = .int64
        XCTAssertThrowsError(try LossTargetMapping(componentID:"loss_bce_with_logits",inputColumns:["x"],roles:wrong).validate())
    }
}
