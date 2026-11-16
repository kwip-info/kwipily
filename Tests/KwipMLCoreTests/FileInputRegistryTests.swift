import XCTest
@testable import KwipMLCore

final class FileInputRegistryTests: XCTestCase {
    private var root: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    func testPortableSourceHasOneTablePortAndCompleteReaderDefaults() throws {
        let registry = try DataFlowOperatorRegistry.load(from: root.appendingPathComponent("runner/dataflow-operators.json"))
        let component = try XCTUnwrap(registry.operatorDefinition(id: DataFlowSourceOperators.fileInput))
        XCTAssertEqual(component.title, "File Input")
        XCTAssertTrue(component.inputs.isEmpty)
        XCTAssertEqual(component.outputs.map(\.id), ["table"])
        XCTAssertEqual(component.outputs.map(\.kind), ["table"])
        XCTAssertEqual(component.backend, "portable")
        XCTAssertEqual(try FileInputConfiguration.validate(component.defaultConfig), FileInputConfiguration.newDefaults)
        XCTAssertFalse(component.isCanvasSink)
    }
}
