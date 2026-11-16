import XCTest
@testable import KwipMLCore

final class CanvasPortDescriptorTests:XCTestCase {
    func testTensorDeclarationIsNotMeasuredAndResolvesOnlyIntegralDimensions() {
        let port=PortContract("in",direction:.input,schema:.tensor(TensorContract([.symbol("batch"),.parameter("width")])),phases:[.forward])
        let value=CanvasPortDescriptor.contract(port,configuration:["width":.integer(6)])
        XCTAssertEqual(value.kind,.tensor);XCTAssertEqual(value.glyph,.circle)
        XCTAssertTrue(value.declared.contains("[batch, 6]"));XCTAssertTrue(value.declared.contains("float32"))
        XCTAssertTrue(value.measured.isEmpty);XCTAssertTrue(value.accessibilityDescription.contains("Declared"))
        XCTAssertTrue(CanvasPortDescriptor.contract(port,configuration:["width":.number(2.5)]).declared.contains("[batch, width unresolved]"))
    }
    func testStaticDeclarationsMatchPinnedBridgeWithoutClaimingRuntime() throws {
        let registry=try ComponentRegistry.builtIn()
        XCTAssertEqual(try BuiltInCapabilityBridge.declarations(registry:registry),try BuiltInCapabilityBridge.package(registry:registry,runtimeVersion:"2.14.1").capabilities)
        XCTAssertThrowsError(try BuiltInCapabilityBridge.package(registry:registry,runtimeVersion:"declaration-only"))
    }
    func testMalformedSchemaRemainsUnknown() {
        let empty=PortContract("p",direction:.input,schema:PortSchema(kind:.table),phases:[])
        XCTAssertEqual(CanvasPortDescriptor.contract(empty).kind,.unknown)
    }
    func testWorkerPortSemanticsAndUnknownNeverGuess() throws {
        func port(_ json:String)throws->DataFlowOperatorPort {try JSONDecoder().decode(DataFlowOperatorPort.self,from:Data(json.utf8))}
        let training=CanvasPortDescriptor.data(try port(#"{"id":"training","kind":"table","partition":"training","required":true}"#),direction:.output)
        XCTAssertEqual(training.kind,.table);XCTAssertEqual(training.partition,.training);XCTAssertEqual(training.glyph,.square)
        let state=CanvasPortDescriptor.data(try port(#"{"id":"state","kind":"state","stateKind":"fittedTransform","required":true}"#),direction:.input)
        XCTAssertEqual(state.kind,.fittedState);XCTAssertEqual(state.glyph,.diamond)
        let unknown=CanvasPortDescriptor.data(try port(#"{"id":"in","kind":"future","required":true}"#),direction:.input)
        XCTAssertEqual(unknown.kind,.unknown);XCTAssertEqual(unknown.glyph,.question);XCTAssertTrue(unknown.measured.isEmpty)
    }
}
