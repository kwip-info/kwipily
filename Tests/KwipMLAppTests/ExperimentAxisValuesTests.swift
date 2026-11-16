import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ExperimentAxisValuesTests: XCTestCase {
    func testIndividualValuesRoundTripTypedAxesAndPreserveOrderAndBudget() throws {
        let original = WorkflowExperimentSpecification(axes: [
            .init(field: "seed", values: [.integer(42), .integer(7)]),
            .init(field: "learningRate", values: [.number(0.01), .number(0.1)]),
            .init(field: "device", values: [.string("cpu")])
        ], budget: .init(maximumRuns: 8, maximumTotalEpochs: 50))
        let values = ExperimentAxisValuesDraft(specification: original)
        XCTAssertEqual(values.values["seed"], ["42", "7"])
        try values.validate()
        let draft = values.applying(to: .init(specification: original))
        XCTAssertEqual(try draft.specification(), original)
        XCTAssertEqual(try draft.summary(baseEpochs: 3).runs, 4)
        XCTAssertEqual(try draft.summary(baseEpochs: 3).epochs, 12)
    }
    func testAddEditRemoveControlsChangeOneOptionAtATimeWithoutHiddenCSVExpansion() throws {
        var values = ExperimentAxisValuesDraft()
        values.add("batchSize"); values.set("4", key: "batchSize", index: 0)
        values.add("batchSize"); values.set("8", key: "batchSize", index: 1)
        try values.validate()
        XCTAssertEqual(try values.applying(to: .init()).specification().axes.first?.values, [.integer(4), .integer(8)])
        values.set("8,16", key: "batchSize", index: 1)
        XCTAssertThrowsError(try values.validate(), "A pasted comma must not silently create two combinations")
        values.set("16", key: "batchSize", index: 1)
        values.remove("batchSize", index: 0)
        try values.validate()
        XCTAssertEqual(values.values["batchSize"], ["16"])
        values.remove("batchSize", index: 0)
        XCTAssertNil(values.values["batchSize"])
        XCTAssertThrowsError(try values.applying(to: .init()).specification(), "Clear explicitly removes an experiment; an empty grid cannot become a hidden single run")
    }
    func testEmptyDuplicateOutOfRangeAndValueCountRemainBounded() throws {
        var values = ExperimentAxisValuesDraft()
        values.add("seed")
        XCTAssertThrowsError(try values.validate())
        values.set("-1", key: "seed", index: 0)
        try values.validate()
        XCTAssertThrowsError(try values.applying(to: .init()).specification())
        values.set("42", key: "seed", index: 0)
        values.add("seed"); values.set("42", key: "seed", index: 1)
        XCTAssertThrowsError(try values.applying(to: .init()).specification())
        for _ in 0..<40 { values.add("seed") }
        XCTAssertEqual(values.values["seed"]?.count, 32)
        values.add("unsupported")
        XCTAssertNil(values.values["unsupported"])
    }
}
