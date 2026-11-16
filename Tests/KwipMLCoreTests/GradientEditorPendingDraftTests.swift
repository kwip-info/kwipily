import XCTest
@testable import KwipMLCore

final class GradientEditorPendingDraftTests: XCTestCase {
    func testOnlyExactActualGradientEditorKeysReceiveUTF8Budget() throws {
        var draft = TensorLearningDraft(projectID: UUID().uuidString, operation: GradientLearningTask.selectedInfluence.rawValue, mode: .practice)
        for (type, field) in [("gradient_policy", "gradientPolicyEditor"), ("trainability", "trainabilityEditor")] {
            let node = try XCTUnwrap(draft.graph.nodes.first { $0.type == type })
            let key = "\(node.id.utf8.count):\(node.id):\(field)"
            draft.pendingFields = [key: String(repeating: "é", count: 65536)]
            XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()), draft)
            XCTAssertThrowsError(try draft.submissionData())
            draft.pendingFields[key] = String(repeating: "é", count: 65537)
            XCTAssertThrowsError(try draft.encoded())
            draft.pendingFields = ["\(node.id.utf8.count):\(node.id):other": String(repeating: "x", count: 513)]
            XCTAssertThrowsError(try draft.encoded())
            draft.pendingFields = [key: String(repeating: "x", count: 513)]
            draft.graph.nodes.removeAll { $0.id == node.id }
            XCTAssertThrowsError(try draft.encoded())
            draft.graph = try GradientLearningExamples.defaultGraph(task: .selectedInfluence)
        }
    }
    func testForeignFamilyAndOldEntryBoundsStayStrict() throws {
        var other = TensorLearningDraft(projectID: UUID().uuidString, operation: "reshape", mode: .practice)
        other.pendingFields = ["15:gradient_policy:gradientPolicyEditor": String(repeating: "x", count: 513)]
        XCTAssertThrowsError(try other.encoded())
        var draft = TensorLearningDraft(projectID: UUID().uuidString, operation: GradientLearningTask.selectedInfluence.rawValue, mode: .practice)
        draft.pendingFields = Dictionary(uniqueKeysWithValues: (0..<65).map { ("field\($0)", "-") })
        XCTAssertThrowsError(try draft.encoded())
    }
}
