import XCTest
import KwipMLCore
@testable import KwipMLApp

final class LegacyDataInspectionBusyTests: XCTestCase {
    @MainActor func testDirectInspectionOwnsSharedBusyBudgetAndCannotBeClearedBySampleAction() {
        let store = WorkspaceStore(rootOverride: FileManager.default.temporaryDirectory, detectCapabilities: false)
        store.graph.execution = ExecutionSettings(datasetCSV: "/selected/source.csv")
        store.dataColumns = ["existing"]; store.isInspectingData = true
        XCTAssertTrue(store.isBusy)
        let before = store.graph
        store.useSampleData()
        XCTAssertTrue(store.isInspectingData)
        XCTAssertEqual(store.graph, before)
        XCTAssertEqual(store.dataColumns, ["existing"])
    }
    @MainActor func testInspectionStartupDoesNotResetEvidenceDuringAnotherWorker() {
        for kind in ["training", "preview", "preparation", "inspection"] {
            let store = WorkspaceStore(rootOverride: FileManager.default.temporaryDirectory, detectCapabilities: false)
            store.dataColumns = ["existing"]; store.dataRows = [["1"]]; store.dataRowCount = 1
            switch kind {
            case "training": store.isRunning = true
            case "preview": store.isPreviewing = true
            case "preparation": store.isPreparingData = true
            default: store.isInspectingData = true
            }
            store.inspectData()
            XCTAssertEqual(store.dataColumns, ["existing"], kind)
            XCTAssertEqual(store.dataRowCount, 1, kind)
        }
    }
}
