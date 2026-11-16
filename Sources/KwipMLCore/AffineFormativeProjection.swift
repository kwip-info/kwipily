import Foundation

/// Deliberately non-Codable: a bounded formative view, not an assessment receipt or handout.
public struct AffineFormativeFixture: Equatable, Identifiable {
    public let id: String
    public let input: LearningTensor
    public let measured: LearningTensor
    public let expected: LearningTensor
    /// Signed measured minus expected, in the same row-major order.
    public let difference: LearningTensor
}

public struct AffineFormativeProjection: Equatable {
    public let fixtures: [AffineFormativeFixture]
    public let submissionDigest: String
    public let absoluteTolerance: Double
    public let relativeTolerance: Double

    /// The caller supplies verified runner evidence/receipt and additionally gates the current UI mode.
    /// Mode here is the recorded submission mode, never the currently selected mode.
    public static func make(plan: LearningAssessmentPlan, evidence: LearningEvidence,
                            recordedMode: LearningMode, lessonIdentity: AffineLessonIdentity = .composition) -> AffineFormativeProjection? {
        guard recordedMode != .assessment,
              plan.id == lessonIdentity.assessmentID,
              plan.fixtures.count == 4, evidence.fixtures.count == 4,
              let reference = try? AffineLessonTasks.plan(
                identity: lessonIdentity, runtimeVersion: plan.runtime.version, fixtureSeed: plan.fixtureSeed,
                assessmentVersion: plan.assessmentVersion),
              plan == reference, plan.criteria.count == 1,
              case .tensor(let key, let absolute, let relative) = plan.criteria[0].rule,
              key == "output" else { return nil }
        func bounded(_ tensor: LearningTensor) -> Bool {
            tensor.shape == [2, 2] && tensor.values.count == 4 && tensor.wellFormed
        }
        guard evidence.fixtures.allSatisfy({ fixture in
            Set(fixture.tensors.keys) == ["output"] && fixture.partitions.isEmpty &&
            fixture.tensors.values.allSatisfy(bounded)
        }), let grade = try? LearningAssessor.evaluate(plan, evidence: evidence),
              grade.status == .passed || grade.status == .failed else { return nil }
        // Assessor established exact unique fixture membership before dictionary construction.
        let observed = Dictionary(uniqueKeysWithValues: evidence.fixtures.map { ($0.fixtureID, $0) })
        var projected: [AffineFormativeFixture] = []
        for fixture in plan.fixtures {
            guard let input = fixture.inputs["x"], let expected = fixture.expected["output"],
                  let measured = observed[fixture.id]?.tensors["output"],
                  bounded(input), bounded(expected) else { return nil }
            let values = zip(measured.values, expected.values).map { $0 - $1 }
            guard values.allSatisfy(\.isFinite) else { return nil }
            projected.append(AffineFormativeFixture(id: fixture.id, input: input, measured: measured,
                expected: expected, difference: LearningTensor(shape: [2, 2], values: values)))
        }
        return AffineFormativeProjection(fixtures: projected, submissionDigest: evidence.submissionDigest,
            absoluteTolerance: absolute, relativeTolerance: relative)
    }
}
