import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class GeluCapabilityActionsTests:XCTestCase {
    private var descriptor:[String:Any] {["version":["major":1,"minor":0,"patch":0],"adapterID":"kwip.runner.graph-v2.gelu","graphSchemaVersions":[2],"approximateTanhModes":[false,true]]}
    private func data(_ value:Any)throws->Data {try JSONSerialization.data(withJSONObject:value,options:.sortedKeys)}
    private func install(_ store:WorkspaceStore,gelu:Data?)throws {
        let contract=try BuiltInCapabilityBridge.executionContract(registry:store.registry,runtimeVersion:"2.14.1",availableDevices:[.cpu])
        try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:JSONEncoder().encode(contract),geluDescriptor:gelu, tensorCompositionDescriptors: fixtureReviewedCompositionDescriptors())
    }
    private func graph(_ mode:Bool)->GraphDocument {
        GraphDocument(nodes:[GraphNode(id:"i",type:"input",config:["features":2]),GraphNode(id:"m",type:"linear",config:["in_features":2,"out_features":2]),GraphNode(id:"g",type:"gelu",config:["approximate_tanh":.bool(mode)]),GraphNode(id:"o",type:"output")],edges:[GraphEdge(source:"i",target:"m"),GraphEdge(source:"m",target:"g"),GraphEdge(source:"g",target:"o")])
    }
    func testLegacyRuntimeOnlyBlocksGeluAcrossRunPreviewAndCodeBothModes()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:directory)}
        let store=WorkspaceStore(rootOverride:root,storageOverride:directory,detectCapabilities:false)
        guard store.runtime != nil else {throw XCTSkip("Development runtime unavailable")}
        let legacy=store.graph
        try install(store,gelu:nil)
        XCTAssertNil(store.capabilityBoundaryError);XCTAssertEqual(store.verifiedRuntimeComponentCount,7)
        for mode in [false,true] {
            store.graph=graph(mode);store.selectedNodeID="g"
            XCTAssertTrue(store.validation.isValid)
            let error=try XCTUnwrap(store.capabilityBoundaryError);XCTAssertTrue(error.contains("GELU"))
            store.run();XCTAssertEqual(store.runError,error);XCTAssertFalse(store.isRunning)
            store.previewSelected();XCTAssertEqual(store.statusMessage,error);XCTAssertFalse(store.isPreviewing)
            store.refreshDraftCode();XCTAssertEqual(store.draftCodeError,error);XCTAssertFalse(store.isGeneratingCode)
            try install(store,gelu:data(descriptor));XCTAssertNil(store.capabilityBoundaryError)
            XCTAssertEqual(store.verifiedRuntimeComponentCount,8)
            try install(store,gelu:nil);XCTAssertNil(store.geluRuntimeCapability)
        }
        store.graph=legacy;XCTAssertNil(store.capabilityBoundaryError)
    }
    func testMalformedOptionalCapabilityDoesNotBreakLegacyButEveryInstallClearsPriorGelu()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:root)}
        let store=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        try install(store,gelu:data(descriptor));XCTAssertNotNil(store.geluRuntimeCapability)
        var malformed=descriptor;malformed["approximateTanhModes"]=[false]
        try install(store,gelu:data(malformed));XCTAssertNil(store.geluRuntimeCapability)
        XCTAssertNil(store.capabilityBoundaryError);XCTAssertEqual(store.verifiedRuntimeComponentCount,7)
        try install(store,gelu:data(descriptor))
        XCTAssertThrowsError(try store.installRuntimeContract(version:"broad",expectedVersion:nil))
        XCTAssertNil(store.geluRuntimeCapability);XCTAssertEqual(store.verifiedRuntimeComponentCount,0)
    }
    func testNewProbeClearsOptionalCapabilityBeforeCompletionAndDoesNotCarryItIntoOldWorker()async throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:root)}
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let store=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let runtime = try XCTUnwrap(store.runtime,store.runtimeError ?? "Configured fixture runtime unavailable")
        let version=runtime.packageVersions["torch"] ?? "2.14.1"
        let contract=try BuiltInCapabilityBridge.executionContract(registry:store.registry,runtimeVersion:version,availableDevices:[.cpu])
        let raw=try JSONSerialization.jsonObject(with:JSONEncoder().encode(contract))
        var event:[String:Any]=["event":"capabilities","torchVersion":version,"devices":["cpu"],"defaultDevice":"cpu","executionContract":raw]
        let payload=String(decoding:try data(event),as:UTF8.self)
        var emit: ((String)->Void)?, finish: ((Int32,String)->Void)?
        store.capabilityLaunch={_,_,_,_,line,exit in emit=line;finish=exit;return .init(cancel:{})}
        try install(store,gelu:data(descriptor))
        var accepted:Bool?
        store.refreshCapabilities { accepted=$0 }
        XCTAssertNil(store.geluRuntimeCapability,"Reprobe must clear old optional support synchronously.")
        XCTAssertNil(accepted,"No capability may install before the retained worker exits.")
        emit?(payload);finish?(0,"")
        XCTAssertEqual(accepted,true,store.capabilityStatus);XCTAssertNil(store.geluRuntimeCapability)
        XCTAssertEqual(store.verifiedRuntimeComponentCount,5);XCTAssertTrue(store.capabilityStatus.contains("5 verified components"))
        XCTAssertTrue(store.capabilityBoundaryError?.contains("Concatenate requires") == true, "Old worker must not inherit prior composition support")
        store.graph=graph(false);XCTAssertNotNil(store.capabilityBoundaryError)
        event["componentCapabilities"]=["gelu":descriptor]
        let updated=String(decoding:try data(event),as:UTF8.self)
        var compatible:Bool?
        store.refreshCapabilities { compatible=$0 }
        emit?(updated);finish?(0,"")
        XCTAssertEqual(compatible,true,store.capabilityStatus);XCTAssertNotNil(store.geluRuntimeCapability)
        XCTAssertEqual(store.verifiedRuntimeComponentCount,6);XCTAssertTrue(store.capabilityStatus.contains("6 verified components"))
        XCTAssertNil(store.capabilityBoundaryError)
    }
}
