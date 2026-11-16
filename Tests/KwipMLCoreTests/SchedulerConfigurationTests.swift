import XCTest
@testable import KwipMLCore

final class SchedulerConfigurationTests: XCTestCase {
    func testFiveCanonicalDefaultsRoundTripAndRejectForeignControls() throws {
        XCTAssertEqual(SchedulerConfiguration.componentIDs.count, 5)
        for id in SchedulerConfiguration.componentIDs {
            let value = try SchedulerConfiguration(componentID: id, values: [:])
            let bytes = try JSONEncoder().encode(value.recipeValue)
            XCTAssertEqual(try SchedulerConfiguration.decodeRecipe(JSONDecoder().decode(JSONValue.self, from: bytes)), value)
            XCTAssertThrowsError(try SchedulerConfiguration(componentID: id, values: ["last_epoch": 1]))
            XCTAssertThrowsError(try SchedulerConfiguration(componentID: id, values: ["foreign": false]))
        }
    }
    func testStrictControlsRequireSafeTypesBoundsAndNonemptyOneCyclePhases() throws {
        for controls: [String: JSONValue] in [["step_size": true], ["step_size": 0], ["gamma": -1]] {
            XCTAssertThrowsError(try SchedulerConfiguration(componentID: "scheduler_step_lr", values: controls))
        }
        for controls: [String: JSONValue] in [["planned_total_batches": 2], ["planned_total_batches": 10, "pct_start": 0.1], ["div_factor": 0], ["anneal_strategy": .string("foreign")], ["cycle_momentum": 1]] {
            XCTAssertThrowsError(try SchedulerConfiguration(componentID: "scheduler_one_cycle_lr", values: controls))
        }
        XCTAssertNoThrow(try SchedulerConfiguration(componentID: "scheduler_one_cycle_lr", values: ["planned_total_batches": 8, "pct_start": 0.5, "cycle_momentum": false]))
        XCTAssertThrowsError(try SchedulerConfiguration(componentID: "scheduler_reduce_on_plateau", values: ["metric": .string("training_loss")]))
        XCTAssertThrowsError(try SchedulerConfiguration(componentID: "scheduler_linear_lr", values: ["start_factor": 0]))
    }
}
