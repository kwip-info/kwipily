import Foundation
import KwipMLCore

extension WorkspaceStore {
    func showLesson(entryID:String) {
        guard ComponentLessonDestination.isAvailable(entryID:entryID) else {return}
        if lessonRailEntryID != entryID {lessonNodeFocus = .init()}
        lessonRailEntryID=entryID;showLessonRail=true
        if let id=activeProjectID {_=lessonReading.associate(entryID:entryID,projectID:id)}
    }
    func restoreLessonRail() {
        guard let id=activeProjectID else {lessonRailEntryID=nil;return}
        if let entry=lessonReading.entry(projectID:id) {lessonRailEntryID=entry;showLessonRail=true}
        else if workflowSession?.owner.document.guidedBuild != nil {showLesson(entryID:"guided")}
        else if tutorialStep != nil {showLesson(entryID:"guided")}
        else if isAffineLessonActive {showLesson(entryID:"affine")}
        else if lessonContext?.kind == .tensor,let operation=learningProjects.tensorDraftSession.draft?.operation,
                let entry=ConstructionLessonCatalog.entryID(operation:operation) {showLesson(entryID:entry)}
        else {lessonRailEntryID=nil}
    }
    func inheritLesson(from source:WorkspaceStore?) {
        guard let entry=source?.launchLessonEntryID ?? source?.lessonRailEntryID else {return}
        showLesson(entryID:entry)
    }
}

@MainActor enum LessonSaveAction {
    @discardableResult static func save(workspace:WorkspaceStore,entryID:String,saveDocument:()->Bool)->Bool {
        guard workspace.lessonRailEntryID==entryID,let projectID=workspace.activeProjectID else {return false}
        guard saveDocument(),workspace.activeProjectID==projectID else {return false}
        guard workspace.lessonReading.saved(entryID:entryID,projectID:projectID) else {
            workspace.statusMessage="Canvas saved; reading progress needs attention.";return false
        }
        workspace.refreshProjects();return true
    }
}
