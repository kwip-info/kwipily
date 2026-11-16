import Foundation
import KwipMLCore

enum LearningOverviewStatus: String, Sendable {
    case saved = "Saved lesson", verified = "Verified", practiced = "Practice checked", inProgress = "Draft saved", needsRecheck = "Needs recheck", unavailable = "Open to check"
    static func classify(passed: Bool, assessment: Bool, matchesDraft: Bool) -> Self {
        guard matchesDraft else { return .needsRecheck }
        guard passed else { return .inProgress }
        return assessment ? .verified : .practiced
    }
}

struct LearningOverviewRecord: Identifiable, Sendable {
    let id: String
    let name: String
    let updatedAt: Date
    let status: LearningOverviewStatus
    static func read(projects: [ProjectRecord], directory: URL, savedLessonIDs:Set<String> = []) -> [Self] {
        let drafts = TensorLearningDraftRepository(directory: directory)
        let receipts = TensorLearningReceiptRepository(directory: directory)
        return projects.sorted { $0.updatedAt > $1.updatedAt }.map { project in
            var status: LearningOverviewStatus = savedLessonIDs.contains(project.id) ? .saved:.unavailable
            // Reading the receipt repository replays and validates measured evidence.
            // A plain visit or progress.json entry cannot become verified completion.
            if let snapshot = try? drafts.load(projectID: project.id) {
                status = .inProgress
                do {
                    if let receipt = try receipts.latest(projectID: project.id) {
                        status = .classify(passed: receipt.attempt.grade.status == .passed,
                            assessment: receipt.attempt.mode == .assessment,
                            matchesDraft: receipt.operation == snapshot.draft.operation && receipt.attempt.mode == snapshot.draft.mode && receipt.submissionData == (try? snapshot.draft.submissionData()))
                    }
                } catch { status = .needsRecheck }
            }
            return Self(id: project.id, name: project.name, updatedAt: project.updatedAt, status: status)
        }
    }
}
