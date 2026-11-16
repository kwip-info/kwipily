import Foundation
import Combine

@MainActor final class LessonReadingStore: ObservableObject {
    enum Status: String { case new = "New", inProgress = "In progress", viewed = "Viewed" }
    struct Record: Codable, Equatable {
        var visited: [String:String] = [:]
        var content:LessonContent? = nil
        var position: String? = nil
        var savedProjectID: String? = nil
        var savedAt: Date? = nil
    }
    private struct Archive: Codable { var version = 1; var records: [String:Record] = [:]; var projects: [String:String] = [:]; var unsavedProjects:[String]? = nil }
    private static var sharedStores: [String:LessonReadingStore] = [:]
    static func shared(directory: URL) -> LessonReadingStore {
        let key=directory.standardizedFileURL.path
        if let store=sharedStores[key] { return store }
        let store=LessonReadingStore(directory:directory);sharedStores[key]=store;return store
    }
    @Published private(set) var revision = 0
    @Published private(set) var error: String?
    private var archive = Archive()
    private let file: URL
    private var writable = true
    init(directory: URL) {
        file=directory.appendingPathComponent("learning/reading.json")
        if FileManager.default.fileExists(atPath:file.path) {
            do {
                let values=try file.resourceValues(forKeys:[.isRegularFileKey,.isSymbolicLinkKey,.fileSizeKey])
                guard values.isRegularFile == true,values.isSymbolicLink != true,(values.fileSize ?? 0)<=2_000_000 else {throw CocoaError(.fileReadCorruptFile)}
                archive=try JSONDecoder().decode(Archive.self,from:Data(contentsOf:file))
                guard archive.version==1 else {throw CocoaError(.fileReadCorruptFile)}
            } catch {self.error="Reading progress could not be opened. Existing records are preserved.";writable=false}
        }
    }
    func record(_ id: String) -> Record {archive.records[id] ?? Record()}
    func entry(projectID: String) -> String? {archive.projects[projectID]}
    func status(_ content: LessonContent) -> Status {
        let record=readingRecord(content)
        let count=content.steps.filter {record.visited[$0.id]==$0.revision}.count
        return count == 0 ? .new : (count == content.steps.count ? .viewed:.inProgress)
    }
    private func readingKey(_ content:LessonContent)->String {content.readingScopeID.map {"task-reading/"+$0} ?? content.entryID}
    private func readingRecord(_ content:LessonContent)->Record {record(readingKey(content))}
    func position(_ content: LessonContent) -> Int {content.steps.firstIndex {$0.id==readingRecord(content).position} ?? 0}
    @discardableResult func visit(_ index: Int, content: LessonContent) -> Bool {
        guard content.steps.indices.contains(index) else {return false}
        var record=readingRecord(content);let step=content.steps[index]
        guard record.position != step.id || record.visited[step.id] != step.revision else {return true}
        record.position=step.id;record.visited[step.id]=step.revision;record.content=content
        return update {archive in
            let key=readingKey(content)
            if key != content.entryID {
                // Retain pre-scope reading evidence once, while keeping the existing bookmark authority.
                let legacyKey="legacy-reading/"+content.entryID
                if let old=archive.records[content.entryID],old.content?.readingScopeID == nil,archive.records[legacyKey] == nil {archive.records[legacyKey]=old}
                archive.records[key]=record
                var latest=archive.records[content.entryID] ?? Record()
                latest.visited=record.visited;latest.position=record.position;latest.content=content
                archive.records[content.entryID]=latest
            } else {archive.records[key]=record}
        }
    }
    @discardableResult func associate(entryID:String,projectID:String) -> Bool {
        guard !entryID.isEmpty,!projectID.isEmpty else {return false}
        if archive.projects[projectID]==entryID{return true}
        return update {$0.projects[projectID]=entryID}
    }
    /// Called only after the existing document owner successfully saves the construction.
    @discardableResult func saved(entryID:String,projectID:String) -> Bool {
        var record=record(entryID);record.savedProjectID=projectID;record.savedAt=Date()
        return update {$0.records[entryID]=record;$0.projects[projectID]=entryID;$0.unsavedProjects?.removeAll {$0==projectID}}
    }
    func isUnsaved(projectID:String)->Bool {archive.unsavedProjects?.contains(projectID) == true}
    /// Remove only the bookmark; the project, reading position and measured attempts remain.
    @discardableResult func unsave(projectID:String)->Bool {
        guard !projectID.isEmpty else {return false}
        return update { archive in
            var hidden=Set(archive.unsavedProjects ?? []);hidden.insert(projectID)
            archive.unsavedProjects=hidden.sorted()
            for id in Array(archive.records.keys) where archive.records[id]?.savedProjectID==projectID {
                archive.records[id]?.savedProjectID=nil;archive.records[id]?.savedAt=nil
            }
        }
    }
    private func update(_ change: (inout Archive)->Void) -> Bool {
        guard writable else {return false}
        var next=archive;change(&next)
        do {
            let data=try JSONEncoder().encode(next)
            guard data.count<=2_000_000 else {throw CocoaError(.fileWriteOutOfSpace)}
            try FileManager.default.createDirectory(at:file.deletingLastPathComponent(),withIntermediateDirectories:true)
            try data.write(to:file,options:.atomic)
            archive=next;error=nil;revision+=1;return true
        } catch {self.error="Reading progress could not be saved: \(error.localizedDescription)";return false}
    }
}
