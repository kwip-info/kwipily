import XCTest
import KwipMLCore
@testable import KwipMLApp

final class MixtureRouterPresentationTests:XCTestCase {
    private func source()throws->String {
        let document=try MixtureExpertsPractice.document()
        guard case .modelPrimitive(_,_,let config)=document.nodes.first(where:{$0.address.localID=="mixture"})?.payload,let text=config["router"]?.stringValue else{throw WorkflowArchiveError("Missing fixture")}
        return text
    }
    func testReadableSourceRoundTripsWithoutScientificChanges()throws {
        let source=try source(),formatted=MixtureRouterPresentation.formatted(source)
        XCTAssertTrue(formatted.contains("\n"))
        XCTAssertEqual(try MixtureRouterPresentation.object(source),try MixtureRouterPresentation.object(MixtureRouterPresentation.compact(formatted)))
        XCTAssertThrowsError(try MixtureRouterPresentation.compact("{\"nodes\":"))
        XCTAssertThrowsError(try MixtureRouterPresentation.compact(String(repeating:" ",count:16385)))
    }
    func testStructuredFieldEditPreservesIdentityWiresAndOtherConfiguration()throws {
        let source=try source(),registry=try ComponentRegistry.builtIn(),before=try MixtureRouterPresentation.graph(source)
        let gate=try XCTUnwrap(before.nodes.first{$0.type=="linear"})
        let changed=try MixtureRouterPresentation.setting("bias",to:false,nodeID:gate.id,source:source,registry:registry)
        let after=try MixtureRouterPresentation.graph(changed)
        XCTAssertEqual(before.edges,after.edges);XCTAssertEqual(before.nodes.map(\.id),after.nodes.map(\.id))
        XCTAssertEqual(after.nodes.first{$0.id==gate.id}?.config["bias"],false)
        XCTAssertEqual(after.nodes.first{$0.id==gate.id}?.config["out_features"],gate.config["out_features"])
        XCTAssertThrowsError(try MixtureRouterPresentation.setting("out_features",to:0,nodeID:gate.id,source:source,registry:registry))
        XCTAssertThrowsError(try MixtureRouterPresentation.setting("bias",to:false,nodeID:"missing",source:source,registry:registry))
    }
}
