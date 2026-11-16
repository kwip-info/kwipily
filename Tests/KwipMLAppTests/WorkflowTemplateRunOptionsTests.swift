import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowTemplateRunOptionsTests:XCTestCase {
    func testExplicitOptionsOnlyChangeTheirProjectAndSettingGroup()throws {
        let root=try canonicalBackendTemporaryDirectory()
        defer {try? FileManager.default.removeItem(at:root)}
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let document=try WorkflowTemplates.instantiate(id:"item-sales-reconciliation")
        _=try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID))
        let owner=try XCTUnwrap(workspace.workflowSession?.owner)
        let entry=try XCTUnwrap(WorkflowTemplates.catalog().first {$0.id=="item-sales-reconciliation"})
        let existing=try ProjectExecutionOverrides(maximumRunSeconds:91,cpuThreads:2)
        XCTAssertTrue(workspace.setProjectExecutionOverrides(existing,projectID:document.projectID))
        let before=try workspace.projectSettingsSnapshot(projectID:document.projectID)
        let otherID=UUID().uuidString
        try workspace.projectExecutionPreferenceStore.save(existing,projectID:otherID)
        let resource=try WorkflowTemplateRunOptionsPlan.applyWorkingSpace(entry,owner:owner,workspace:workspace)
        XCTAssertEqual(resource.overrides.workloadPolicy?.preset,.balanced)
        XCTAssertEqual(resource.overrides.cpuThreads,2)
        XCTAssertEqual(resource.overrides.maximumRunSeconds,91)
        XCTAssertEqual(resource.history,before.history,"Resource choice must not silently change History")
        let compact=try WorkflowTemplateRunOptionsPlan.applyCompactHistory(entry,owner:owner,workspace:workspace)
        XCTAssertEqual(compact.overrides,resource.overrides,"History choice must not replace resource preferences")
        XCTAssertEqual(compact.history,.init(metricsAndDiagnostics:true,componentOutputs:false))
        XCTAssertEqual(compact.global,before.global)
        XCTAssertEqual(try workspace.projectExecutionPreferenceStore.load(projectID:otherID),existing)
        XCTAssertEqual(owner.document,document,"Run options are not graph edits")
        owner.setExecutionBusy(true)
        XCTAssertThrowsError(try WorkflowTemplateRunOptionsPlan.applyWorkingSpace(entry,owner:owner,workspace:workspace))
        XCTAssertThrowsError(try WorkflowTemplateRunOptionsPlan.applyCompactHistory(entry,owner:owner,workspace:workspace))
        owner.setExecutionBusy(false)
        let wrong=try XCTUnwrap(WorkflowTemplates.catalog().first {$0.id=="rental-demand"})
        XCTAssertThrowsError(try WorkflowTemplateRunOptionsPlan.applyWorkingSpace(wrong,owner:owner,workspace:workspace))
        XCTAssertEqual(try workspace.projectSettingsSnapshot(projectID:document.projectID).revision,compact.revision)
    }
}
