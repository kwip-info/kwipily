import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class TrainInlineRecipesInspectorTests: XCTestCase {
    func testMappingOwnCommitRefreshesObjectiveWithoutDiscardingOptimizerDraft() throws {
        let (_,session,directory)=try fixture();defer{try? FileManager.default.removeItem(at:directory)}
        let optimizer=TrainInlineRecipeDraft(session:session,role:"optimizer")
        let objective=TrainInlineRecipeDraft(session:session,role:"objective")
        XCTAssertTrue(optimizer.choose("optimizer_adamw"))
        XCTAssertTrue(optimizer.setField("lr",value:.number(0.003),origin:optimizer.numericOrigin("lr")))
        var next=session.authority.configuration
        next["objective"]=try LossConfiguration(componentID:"loss_cross_entropy",values:[:]).recipeValue
        XCTAssertTrue(session.authority.configure(next,expected:session.authority.configuration))
        XCTAssertTrue(session.isCurrent)
        XCTAssertEqual(objective.componentID,"loss_cross_entropy")
        XCTAssertFalse(objective.hasChanges)
        XCTAssertTrue(optimizer.hasChanges)
        XCTAssertEqual(optimizer.configuration["lr"],.number(0.003))
        XCTAssertTrue(optimizer.apply(),optimizer.rejection ?? "")
        XCTAssertEqual(session.authority.configuration["objective"],next["objective"])
    }
    func testMappingDoesNotOverwriteConflictingObjectiveDraftAndRevertLoadsNewBaseline() throws {
        let (_,session,directory)=try fixture();defer{try? FileManager.default.removeItem(at:directory)}
        let draft=TrainInlineRecipeDraft(session:session,role:"objective")
        XCTAssertTrue(draft.choose("loss_mse"))
        var next=session.authority.configuration
        next["objective"]=try LossConfiguration(componentID:"loss_cross_entropy",values:[:]).recipeValue
        XCTAssertTrue(session.authority.configure(next,expected:session.authority.configuration))
        XCTAssertEqual(draft.componentID,"loss_mse")
        XCTAssertFalse(draft.canApply)
        XCTAssertFalse(draft.apply())
        XCTAssertTrue(draft.revert())
        XCTAssertEqual(draft.componentID,"loss_cross_entropy")
        XCTAssertFalse(draft.hasChanges)
    }
    private var repository: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func fixture() throws -> (WorkflowAuthoringStore, TrainInlineRecipesSession, URL) {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let library = ProjectLibrary(directory: directory)
        let document = try WorkflowExamples.preparedParallelClassifier(device: "cpu")
        _ = try library.saveSource(.workflow(document), id: document.projectID)
        let owner = try WorkflowAuthoringStore(library: library, projectID: document.projectID, modelRegistry: .builtIn(),
            dataRegistry: .load(from: repository.appendingPathComponent("runner/dataflow-operators.json")))
        XCTAssertTrue(owner.select(.init(nodeAddresses: [WorkflowExamples.trainAddress]), lease: owner.lease()))
        let authority = try WorkflowNodeEditingAuthority(owner: owner, address: WorkflowExamples.trainAddress)
        return (owner, .init(authority: authority), directory)
    }
    func testEveryOptimizerSchedulerAndObjectiveOptionHasATypedFieldAndCanonicalEnvelope() throws {
        let (_, session, directory) = try fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        for role in ["optimizer", "scheduler", "objective"] {
            let draft = TrainInlineRecipeDraft(session: session, role: role)
            for id in TrainInlineRecipeDraft.IDs(role) {
                XCTAssertTrue(draft.choose(id), id)
                XCTAssertNil(draft.validationMessage, id)
                XCTAssertEqual(Set(draft.fields.map(\.key)), Set(draft.configuration.keys), "Every \(id) setting needs a typed field")
                guard case .object(let envelope) = try draft.value() else { return XCTFail("Missing \(id) envelope") }
                XCTAssertEqual(envelope["componentID"], .string(id))
                XCTAssertEqual(envelope["schemaVersion"], .integer(1))
                XCTAssertEqual(envelope["configurationVersion"], .integer(1))
                XCTAssertEqual(envelope["config"], .object(draft.configuration))
            }
            draft.releasePendingEdits()
        }
    }
    func testOwnRoleAppliesPreserveOtherStagedRolesAndUndoIsAtomic() throws {
        let (owner, session, directory) = try fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let optimizer = TrainInlineRecipeDraft(session: session, role: "optimizer")
        let scheduler = TrainInlineRecipeDraft(session: session, role: "scheduler")
        XCTAssertTrue(optimizer.choose("optimizer_sgd"))
        XCTAssertTrue(optimizer.setField("lr", value: .number(0.125), origin: optimizer.numericOrigin("lr")))
        XCTAssertTrue(scheduler.choose("scheduler_step_lr"))
        XCTAssertTrue(scheduler.setField("step_size", value: .integer(3), origin: scheduler.numericOrigin("step_size")))
        XCTAssertTrue(owner.hasPendingInspectorEdits)
        let original = session.expected
        let revision = owner.document.revision
        XCTAssertTrue(optimizer.apply(), optimizer.rejection ?? "")
        XCTAssertEqual(owner.document.revision, revision + 1)
        XCTAssertTrue(session.isCurrent)
        XCTAssertTrue(scheduler.hasChanges, "An optimizer Apply must retain the unapplied scheduler")
        XCTAssertEqual(scheduler.configuration["step_size"], .integer(3))
        XCTAssertNil(session.expected["scheduler"])
        XCTAssertTrue(owner.hasPendingInspectorEdits)
        XCTAssertTrue(scheduler.apply(), scheduler.rejection ?? "")
        XCTAssertEqual(owner.document.revision, revision + 2)
        XCTAssertNotNil(session.expected["optimizer"])
        XCTAssertNotNil(session.expected["scheduler"])
        XCTAssertEqual(session.expected.filter { !["optimizer", "scheduler"].contains($0.key) }, original)
        XCTAssertFalse(owner.hasPendingInspectorEdits)
        XCTAssertTrue(owner.undo(lease: owner.lease()))
        XCTAssertFalse(session.isCurrent)
        let train = try XCTUnwrap(owner.document.nodes.first { $0.address == WorkflowExamples.trainAddress })
        guard case .train(let afterUndo) = train.payload else { return XCTFail("Missing Train") }
        XCTAssertNotNil(afterUndo["optimizer"])
        XCTAssertNil(afterUndo["scheduler"])
        XCTAssertFalse(scheduler.apply(), "Undo retires the old editor")
    }
    func testFamilySwitchRevertPendingNumbersAndStaleSourceAreSafe() throws {
        let (owner, session, directory) = try fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let draft = TrainInlineRecipeDraft(session: session, role: "optimizer")
        XCTAssertTrue(draft.choose("optimizer_sgd"))
        XCTAssertTrue(draft.setField("lr", value: .number(0.25), origin: draft.numericOrigin("lr")))
        XCTAssertTrue(draft.choose("optimizer_adam"))
        XCTAssertTrue(draft.choose("optimizer_sgd"))
        XCTAssertEqual(draft.configuration["lr"], .number(0.25))
        let oldOrigin = draft.numericOrigin("lr")
        draft.setNumericPending("lr", origin: oldOrigin, pending: true)
        XCTAssertFalse(draft.canApply)
        XCTAssertFalse(draft.choose("optimizer_adam"))
        XCTAssertTrue(draft.revert())
        XCTAssertNil(draft.componentID)
        XCTAssertFalse(owner.hasPendingInspectorEdits)
        XCTAssertTrue(draft.choose("optimizer_sgd"))
        XCTAssertFalse(draft.setField("lr", value: .number(0.5), origin: oldOrigin), "A reset invalidates old field callbacks")
        XCTAssertNotNil(owner.apply(.rename(address: WorkflowExamples.trainAddress, name: "Externally changed"), lease: owner.lease()))
        XCTAssertFalse(session.isCurrent)
        XCTAssertFalse(draft.apply())
        XCTAssertNil(owner.document.nodes.first { $0.address == WorkflowExamples.trainAddress }.flatMap { node -> JSONValue? in
            guard case .train(let values) = node.payload else { return nil }; return values["optimizer"]
        })
    }
    func testStructuredEditorsUseRealModelOwnersAndExposeAllThreeRoles() throws {
        let (_, session, directory) = try fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let choices = session.modelChoices()
        XCTAssertFalse(choices.owners.isEmpty)
        XCTAssertTrue(choices.endpoints.contains { $0.isModelInput })
        let initialization = TrainInlineRecipeDraft(session: session, role: "initialization")
        XCTAssertTrue(initialization.choose("initialization"))
        let initialEditor = try XCTUnwrap(initialization.initialization)
        XCTAssertFalse(initialization.canApply, "Starting values require an explicit actual learned-array rule")
        XCTAssertTrue(initialEditor.addOwner())
        XCTAssertTrue(initialization.canApply, initialization.validationMessage ?? "")
        XCTAssertTrue(initialization.apply(), initialization.rejection ?? "")
        let gradient = TrainInlineRecipeDraft(session: session, role: "gradientPolicy")
        XCTAssertTrue(gradient.choose("gradient_policy"))
        let gradientEditor = try XCTUnwrap(gradient.gradient)
        XCTAssertTrue(gradientEditor.setAccumulationSteps(2))
        XCTAssertTrue(gradient.apply(), gradient.validationMessage ?? gradient.rejection ?? "")
        let trainability = TrainInlineRecipeDraft(session: session, role: "trainability")
        XCTAssertTrue(trainability.choose("trainability"))
        let trainabilityEditor = try XCTUnwrap(trainability.trainability)
        XCTAssertTrue(trainabilityEditor.addOwner(try XCTUnwrap(choices.owners.first).id))
        XCTAssertTrue(trainabilityEditor.setMode("frozen", at: 0))
        XCTAssertTrue(trainability.apply(), trainability.validationMessage ?? trainability.rejection ?? "")
        XCTAssertEqual(Set(session.expected.keys).intersection(TrainInlineRecipesSession.roles), ["initialization", "gradientPolicy", "trainability"])
    }
    func testObjectiveScalarizationAppliesWithObjectiveAndRevertRestoresItsDraft() throws {
        let (owner, session, directory) = try fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let objective = TrainInlineRecipeDraft(session: session, role: "objective")
        XCTAssertTrue(objective.choose("loss_cross_entropy"))
        XCTAssertTrue(objective.setField("reduction", value: .string("none"), origin: objective.numericOrigin("reduction")))
        XCTAssertTrue(objective.setObjectiveScalarization("sum"))
        let revision = owner.document.revision
        XCTAssertTrue(objective.apply(), objective.rejection ?? "")
        XCTAssertEqual(owner.document.revision, revision + 1)
        XCTAssertEqual(session.expected["objectiveScalarization"], .string("sum"))
        guard case .object(let envelope) = session.expected["objective"], case .object(let values) = envelope["config"] else { return XCTFail("Missing objective") }
        XCTAssertEqual(values["reduction"], .string("none"))
        XCTAssertTrue(objective.setObjectiveScalarization("mean"))
        XCTAssertTrue(objective.hasChanges)
        XCTAssertTrue(objective.revert())
        XCTAssertEqual(objective.objectiveScalarization, "sum")
        XCTAssertFalse(objective.hasChanges)
        let optimizer = TrainInlineRecipeDraft(session: session, role: "optimizer")
        XCTAssertTrue(optimizer.choose("optimizer_sgd")); XCTAssertTrue(optimizer.apply())
        XCTAssertEqual(session.expected["objectiveScalarization"], .string("sum"))
    }

    func testUnrelatedCommandsThroughSameAuthorityRenewBaseWithoutLosingRoleDraft() throws {
        let (_, session, directory) = try fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let optimizer = TrainInlineRecipeDraft(session: session, role: "optimizer")
        XCTAssertTrue(optimizer.choose("optimizer_sgd"))
        XCTAssertTrue(optimizer.setField("lr", value: .number(0.5), origin: optimizer.numericOrigin("lr")))
        var next = session.authority.configuration
        next["datasetAccess"] = .object(DatasetConfiguration.defaults)
        XCTAssertTrue(session.authority.configure(next, expected: session.authority.configuration))
        XCTAssertTrue(session.isCurrent)
        XCTAssertEqual(optimizer.configuration["lr"], .number(0.5))
        XCTAssertTrue(optimizer.apply(), optimizer.rejection ?? "")
        XCTAssertEqual(session.expected["datasetAccess"], .object(DatasetConfiguration.defaults))
    }

    func testDefaultRemovesOnlySelectedOptionalRole() throws {
        let (_, session, directory) = try fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let optimizer = TrainInlineRecipeDraft(session: session, role: "optimizer")
        XCTAssertTrue(optimizer.choose("optimizer_sgd")); XCTAssertTrue(optimizer.apply())
        let gradient = TrainInlineRecipeDraft(session: session, role: "gradientPolicy")
        XCTAssertTrue(gradient.choose("gradient_policy")); XCTAssertTrue(gradient.apply())
        let preserved = session.expected["gradientPolicy"]
        XCTAssertTrue(optimizer.choose(nil)); XCTAssertTrue(optimizer.apply())
        XCTAssertNil(session.expected["optimizer"])
        XCTAssertEqual(session.expected["gradientPolicy"], preserved)
    }
}
