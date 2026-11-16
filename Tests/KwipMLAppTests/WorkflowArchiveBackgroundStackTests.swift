import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

/// The native History actor uses small cooperative-worker stacks. Ordinary test
/// methods run on much larger stacks and previously concealed this regression.
final class WorkflowArchiveBackgroundStackTests:XCTestCase {
    private let descriptor = try! JSONDecoder().decode(JSONValue.self, from: Data(#"{"version":"1.0.0","adapterID":"kwip.runner.module-composition","graphSchemaVersions":[2],"planSchemaVersion":1,"modelAuthoringSchemaVersion":2,"compositeSourceSchemaVersion":3,"nativeExpansionContractVersion":2,"workflowSchemaVersion":11,"compilerContractVersion":10,"captureFormatVersion":6,"devices":["cpu","mps"],"modes":["branched-structure","nested-structure","parameter-ownership","train-eval","architecture-export","sequential-structure","ordered-structure","named-structure"],"sourceMembers":["torch.nn.Module","torch.nn.Sequential","torch.nn.ModuleList","torch.nn.ModuleDict"],"containerKinds":["graph","sequential","orderedList","namedMap"],"maximumDefinitions":64,"maximumExpandedNodes":128,"maximumEdges":256,"maximumContainers":129,"maximumChildReferences":256,"maximumBoundaryEndpoints":4096,"maximumDepth":8,"maximumPlanBytes":131072,"maximumSourceBytes":2000000,"maximumModelSourceBytes":2000000,"maximumParameters":200000,"parameterOwnership":"expandedNodeRelativeNativeParameter","stateOwnership":"bijectiveNativePersistentState","sharedParameters":false,"sequentialExecution":"nativeForwardValidatedChain","collectionExecution":"explicitDAG","sourceVerification":"independentCapturedRegistryReExpansion","exactResumeDevices":["cpu"],"fineTuneDevices":["cpu","mps"],"mpsResume":"weightsOnly","portableExport":"selfContainedNativeHierarchy","legacyAbsence":"unchanged"}"#.utf8))
    private let pin=String(repeating:"a",count:64)
    private func capture(packed:Bool,weighted:Bool,objective:Bool)throws->WorkflowArchiveCapture {
        let fixtures=BagWorkflowArchiveTests()
        let plan=try fixtures.compile(fixtures.fixture(packed:packed,weighted:weighted,objective:objective))
        func file(_ path:String)->WorkflowArchiveFile {.init(path:path,bytes:10,sha256:pin)}
        return .init(schemaVersion:12,compilerContractVersion:11,projectID:plan.projectID,directoryID:UUID().uuidString,runID:UUID().uuidString,requestIdentity:pin,
            revision:plan.revision,train:plan.train,workflow:file("workflow.json"),modelRegistry:file("model-registry.json"),dataRegistry:file("data-registry.json"),
            modelRegistrySHA256:pin,dataRegistrySHA256:pin,runtimeEvidence:file("runtime-evidence.json"),
            sources:plan.sourceAddresses.enumerated().map {.init(address:$0.element,path:"inputs/source-\($0.offset).csv",bytes:10,sha256:pin)},
            plan:try WorkflowArchivePlan(plan,compilerContractVersion:11),captureFormatVersion:7,moduleCompositionDescriptor:descriptor)
    }
    private final class Reply:@unchecked Sendable {
        private let lock=NSLock()
        private var stored:Result<Data,Error>?
        func put(_ value:Result<Data,Error>) {lock.lock();defer {lock.unlock()};stored=value}
        func get()->Result<Data,Error>? {lock.lock();defer {lock.unlock()};return stored}
    }
    private func decodeOnWorker(_ bytes:Data)throws->Data {
        let completion=expectation(description:"512KB archive decode completed"),reply=Reply()
        let worker=Thread {
            reply.put(Result {try WorkflowArchiveCodec.encodeCapture(WorkflowArchiveCodec.decodeCapture(bytes))})
            completion.fulfill()
        }
        worker.name="KwipML archive 512KB regression"
        worker.stackSize=512*1024
        XCTAssertEqual(worker.stackSize,512*1024)
        worker.start()
        wait(for:[completion],timeout:10)
        return try XCTUnwrap(reply.get()).get()
    }
    func testRealBagCapturesDecodeOn512KBWorkerAndRetainExactBytes()throws {
        for (packed,weighted,objective) in [(false,false,false),(true,true,false),(true,true,true)] {
            let original=try WorkflowArchiveCodec.encodeCapture(capture(packed:packed,weighted:weighted,objective:objective))
            XCTAssertEqual(try decodeOnWorker(original),original)
        }
    }
    func testSmallStackStillRejectsTamperedPacketVersionAndUnknownFields()throws {
        let bytes=try WorkflowArchiveCodec.encodeCapture(capture(packed:true,weighted:true,objective:true))
        let original=try XCTUnwrap(JSONSerialization.jsonObject(with:bytes) as? [String:Any])
        for mutation in ["packet","version","unknown"] {
            var value=original
            if mutation == "version" {value["compilerContractVersion"]=10}
            else if mutation == "unknown" {value["unrecognizedEvidence"]=true}
            else {
                var plan=try XCTUnwrap(value["plan"] as? [String:Any])
                var mapping=try XCTUnwrap(plan["bagInput"] as? [String:Any])
                mapping.removeValue(forKey:"weightColumns");plan["bagInput"]=mapping;value["plan"]=plan
            }
            let malformed=try JSONSerialization.data(withJSONObject:value,options:[.sortedKeys])
            XCTAssertThrowsError(try decodeOnWorker(malformed),mutation)
        }
    }
    @MainActor func testActualCompletedBagHistoryReopensOn512KBWorker()async throws {
        let temporary=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:temporary,withIntermediateDirectories:true)
        guard let physical=realpath(temporary.path,nil) else {throw WorkflowFileLeaseError.unsafePath}
        let root=URL(fileURLWithPath:String(cString:physical));free(physical)
        defer {try? FileManager.default.removeItem(at:root)}
        // Existing actual managed worker fixture: copied/pinned source, guarded
        // discovery, two training epochs and a completed, owned archive.
        let saved=try await BagWorkflowSessionTests().capture(objective:false,root:root)
        let reference=saved.reference
        let completion=expectation(description:"512KB actual History replay completed"),reply=Reply()
        let worker=Thread {
            switch WorkflowHistoryReader.load(reference,root:root) {
            case .verified(let value):
                if value.modelContent?.bagPredictionPreview != nil {
                    reply.put(.success(Data("verified packet history".utf8)))
                } else {reply.put(.failure(WorkflowArchiveError("Completed grouped History lost its packet preview.")))}
            case .unavailable(let message),.incomplete(let message):reply.put(.failure(WorkflowArchiveError(message)))
            case .legacyUnverifiable(_,let message):reply.put(.failure(WorkflowArchiveError(message)))
            }
            completion.fulfill()
        }
        worker.name="KwipML actual History 512KB regression";worker.stackSize=512*1024
        XCTAssertEqual(worker.stackSize,512*1024)
        worker.start()
        await fulfillment(of:[completion],timeout:15)
        XCTAssertEqual(try XCTUnwrap(reply.get()).get(),Data("verified packet history".utf8))
    }

}
