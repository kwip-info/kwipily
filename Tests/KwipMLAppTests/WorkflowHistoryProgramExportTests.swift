import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowHistoryProgramExportTests:XCTestCase {
    private func fixture()throws->(WorkflowHistoryProgramExportSelection,RuntimeLocation,ProgramExportRuntimeCapability) {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:repository,allowDevelopment:true)
        let reference=WorkflowHistoryReference(projectID:UUID().uuidString,directoryID:UUID())
        let selected=WorkflowHistoryProgramExportSelection(reference:reference,identity:.init(runID:UUID().uuidString,requestIdentity:String(repeating:"a",count:64),modelManifestSHA256:String(repeating:"b",count:64)),completed:true,hasModel:true,isBag:false)
        let capability=ProgramExportRuntimeCapability(descriptor:try WorkflowArchiveCodec.encode(BundledProgramExportCapabilities.entry.runtimeContract))
        XCTAssertTrue(capability.isAvailable)
        return(selected,runtime,capability)
    }
    func testVerifiedSavedOrdinarySelectionRequiresIndependentQualifiedRuntime()throws {
        let (selected,runtime,capability)=try fixture()
        XCTAssertNoThrow(try WorkflowHistoryProgramExportFactory.validate(selected,currentReference:selected.reference,active:true,busy:false,loading:false,capability:capability,runtimeVersion:"2.14.1",runtime:runtime))
        XCTAssertThrowsError(try WorkflowHistoryProgramExportFactory.validate(selected,currentReference:selected.reference,active:true,busy:false,loading:false,capability:.init(),runtimeVersion:"2.14.1",runtime:runtime))
        XCTAssertThrowsError(try WorkflowHistoryProgramExportFactory.validate(selected,currentReference:selected.reference,active:true,busy:false,loading:false,capability:capability,runtimeVersion:nil,runtime:runtime))
        XCTAssertThrowsError(try WorkflowHistoryProgramExportFactory.validate(selected,currentReference:selected.reference,active:true,busy:false,loading:false,capability:capability,runtimeVersion:"2.14.1",runtime:nil))
    }
    func testForeignRetiredBusyOrUnverifiedModelCannotOpenAction()throws {
        let (selected,runtime,capability)=try fixture()
        XCTAssertNoThrow(try WorkflowHistoryProgramExportFactory.validate(selected,currentReference:selected.reference,active:true,busy:false,loading:false,capability:capability,runtimeVersion:"2.14.1",runtime:runtime))
        for (active,busy,loading,reference) in [(false,false,false,selected.reference),(true,true,false,selected.reference),(true,false,true,selected.reference),(true,false,false,WorkflowHistoryReference(projectID:UUID().uuidString,directoryID:UUID()))] {
            XCTAssertThrowsError(try WorkflowHistoryProgramExportFactory.validate(selected,currentReference:reference,active:active,busy:busy,loading:loading,capability:capability,runtimeVersion:"2.14.1",runtime:runtime))
        }
        for (completed,hasModel,isBag) in [(false,true,false),(true,false,false),(true,true,true)] {
            let wrong=WorkflowHistoryProgramExportSelection(reference:selected.reference,identity:selected.identity,completed:completed,hasModel:hasModel,isBag:isBag)
            XCTAssertThrowsError(try WorkflowHistoryProgramExportFactory.validate(wrong,currentReference:wrong.reference,active:true,busy:false,loading:false,capability:capability,runtimeVersion:"2.14.1",runtime:runtime))
        }
    }
}
