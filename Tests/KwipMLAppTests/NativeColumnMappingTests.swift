import XCTest
@testable import KwipMLApp
import KwipMLCore
final class NativeColumnMappingTests:XCTestCase {
    @MainActor func testNativeArgumentsUseLosslessJSONHeaderArray()throws {
        let dir=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:dir)}
        let store=WorkspaceStore(rootOverride:dir,detectCapabilities:false)
        store.graph.execution=ExecutionSettings(datasetCSV:"/private/source.csv",featureColumns:[" x","y,unit"],targetColumn:"label")
        let args=try store.trainingDataArguments()
        let index=try XCTUnwrap(args.firstIndex(of:"--feature-columns-json"))
        XCTAssertFalse(args.contains("--feature-columns"))
        XCTAssertEqual(try JSONDecoder().decode([String].self,from:Data(args[index+1].utf8)),[" x","y,unit"])
    }
}
