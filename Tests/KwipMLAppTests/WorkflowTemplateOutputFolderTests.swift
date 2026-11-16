import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowTemplateOutputFolderTests: XCTestCase {
    private var workspaces:[WorkspaceStore]=[]
    private func fixture() throws -> (URL, WorkspaceStore, WorkflowWorkspaceSession, Set<WorkflowNodeAddress>) {
        let root = FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: root, detectCapabilities: false)
        workspaces.append(workspace)
        let document = try WorkflowTemplates.instantiate(id: "sales-exceptions")
        _ = try workspace.projectLibrary.saveSource(.workflow(document), id: document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id: document.projectID), workspace.statusMessage)
        let host = try XCTUnwrap(workspace.workflowSession)
        let addresses = Set(WorkflowTemplateSetupPlan.steps(in: host.owner.document).filter { $0.action == .chooseOutput }.map(\.address))
        XCTAssertEqual(addresses.count, 3)
        XCTAssertTrue(host.owner.select(.init(nodeAddresses: addresses), lease: host.owner.lease()))
        return (root, workspace, host, addresses)
    }
    func testOneFreshPickerBindsOnlyTheSelectedReportsAtomicallyAndSurvivesReload() throws {
        let (root, workspace, host, addresses) = try fixture()
        let before = host.owner.document
        let folder = root.appendingPathComponent("reports")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        var picks = 0
        host.outputDestinations.pickDirectory = { picks += 1; return folder }
        XCTAssertTrue(host.outputDestinations.chooseDestinations(for: addresses, lease: host.owner.lease()), host.outputDestinations.statusMessage)
        XCTAssertEqual(picks, 1)
        XCTAssertEqual(host.outputDestinations.revision, 1, "One private atomic publication, not one mutation per output.")
        XCTAssertEqual(host.owner.document, before)
        let expected = folder.path.hasPrefix("/var/") ? "/private" + folder.path : folder.path
        for address in addresses { XCTAssertEqual(host.outputDestinations.path(for: address), expected) }
        XCTAssertNil(host.outputDestinations.path(for: .init(domain: .data, localID: "source")))
        let restored = try DataOutputDestinationProvider(workspace: workspace, owner: host.owner)
        let access = try restored.acquire(addresses); defer { access.close() }
        XCTAssertEqual(Set(access.paths.values), [expected])
    }
    func testSymlinkFolderCannotPublishAnyReportGrant() throws {
        let (root, _, host, addresses) = try fixture()
        let real = root.appendingPathComponent("real"), link = root.appendingPathComponent("link")
        try FileManager.default.createDirectory(at: real, withIntermediateDirectories: false)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: real)
        host.outputDestinations.pickDirectory = { link }
        XCTAssertFalse(host.outputDestinations.chooseDestinations(for: addresses, lease: host.owner.lease()))
        XCTAssertEqual(host.outputDestinations.revision, 0)
        for address in addresses { XCTAssertNil(host.outputDestinations.path(for: address)) }
    }
    func testChangedOwnerWhilePickerIsOpenCannotPublishAnyOfTheBindings() throws {
        let (root, _, host, addresses) = try fixture()
        host.outputDestinations.pickDirectory = {
            XCTAssertTrue(host.owner.rename("Changed while choosing", lease: host.owner.lease()))
            return root
        }
        XCTAssertFalse(host.outputDestinations.chooseDestinations(for: addresses, lease: host.owner.lease()))
        XCTAssertEqual(host.outputDestinations.revision, 0)
        for address in addresses { XCTAssertNil(host.outputDestinations.path(for: address)) }
    }
    func testCancelWrongSelectionAndNonOutputNeverCreateAuthority() throws {
        let (_, _, host, addresses) = try fixture()
        var picks = 0
        host.outputDestinations.pickDirectory = { picks += 1; return nil }
        XCTAssertFalse(host.outputDestinations.chooseDestinations(for: addresses, lease: host.owner.lease()))
        XCTAssertEqual(picks, 1)
        XCTAssertTrue(host.owner.select(.init(nodeAddresses: [try XCTUnwrap(addresses.first)]), lease: host.owner.lease()))
        XCTAssertFalse(host.outputDestinations.chooseDestinations(for: addresses, lease: host.owner.lease()))
        XCTAssertEqual(picks, 1)
        XCTAssertFalse(host.outputDestinations.chooseDestinations(for: [.init(domain: .data, localID: "source")], lease: host.owner.lease()))
        XCTAssertEqual(picks, 1)
        XCTAssertEqual(host.outputDestinations.revision, 0)
    }
}
