import XCTest
@testable import KwipMLCore

final class OptimizerConfigurationTests: XCTestCase {
    func testCanonicalDefaultsAndTypedRecipe() throws {
        for (id, rate) in [("optimizer_sgd", 0.001), ("optimizer_adam", 0.001), ("optimizer_adamw", 0.001), ("optimizer_rmsprop", 0.01), ("optimizer_adagrad", 0.01)] {
            let config = try OptimizerConfiguration(componentID: id, values: [:])
            XCTAssertEqual(config.learningRate, rate)
            XCTAssertEqual(config.componentID, id)
            XCTAssertEqual(try OptimizerConfiguration(componentID: id, values: config.values), config)
            guard case .object(let recipe) = config.recipeValue else { return XCTFail("Recipe must be a typed object") }
            XCTAssertEqual(recipe["schemaVersion"], .integer(1))
            XCTAssertEqual(recipe["config"], .object(config.values))
        }
        XCTAssertEqual(try OptimizerConfiguration(componentID: "optimizer_adamw", values: [:]).values["weight_decay"], .number(0.01))
    }
    func testNondefaultAndZeroRateArePreserved() throws {
        let sgd = try OptimizerConfiguration(componentID: "optimizer_sgd", values: ["lr": 0, "momentum": 0.8, "nesterov": true])
        XCTAssertEqual(sgd.learningRate, 0)
        XCTAssertEqual(sgd.values["momentum"], .number(0.8))
        XCTAssertEqual(sgd.values["nesterov"], .bool(true))
        let adam = try OptimizerConfiguration(componentID: "optimizer_adam", values: ["beta1": 0.5, "beta2": 0.75, "amsgrad": true, "decoupled_weight_decay": true, "maximize": true])
        XCTAssertEqual(adam.values["beta2"], .number(0.75))
        XCTAssertEqual(adam.values["maximize"], .bool(true))
    }
    func testStrictConfigurationAndCrossFieldBoundaries() throws {
        for id in OptimizerConfiguration.componentIDs {
            for values: [String: JSONValue] in [["lr": true], ["lr": -0.1], ["lr": .number(.infinity)], ["lr": .number(1e39)], ["weight_decay": -1], ["foreach": true], ["config": .object([:])], ["maximize": 1]] {
                XCTAssertThrowsError(try OptimizerConfiguration(componentID: id, values: values))
            }
        }
        for values: [String: JSONValue] in [["nesterov": true], ["momentum": 1.1], ["momentum": 0.5, "dampening": 0.1, "nesterov": true]] {
            XCTAssertThrowsError(try OptimizerConfiguration(componentID: "optimizer_sgd", values: values))
        }
        for id in ["optimizer_adam", "optimizer_adamw"] {
            for values: [String: JSONValue] in [["beta1": 1], ["beta2": -0.1], ["eps": 0], ["eps": .number(1e-40)], ["amsgrad": 1]] {
                XCTAssertThrowsError(try OptimizerConfiguration(componentID: id, values: values))
            }
        }
        XCTAssertThrowsError(try OptimizerConfiguration(componentID: "optimizer_adamw", values: ["decoupled_weight_decay": true]))
        XCTAssertThrowsError(try OptimizerConfiguration(componentID: "optimizer_rmsprop", values: ["alpha": 1.1]))
        XCTAssertThrowsError(try OptimizerConfiguration(componentID: "optimizer_adagrad", values: ["initial_accumulator_value": -1]))
        XCTAssertThrowsError(try OptimizerConfiguration(componentID: "other", values: [:]))
    }
}
