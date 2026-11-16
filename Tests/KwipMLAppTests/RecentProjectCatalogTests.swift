import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class RecentProjectCatalogTests:XCTestCase {
    func testOneLocationAppearsOnceAndManagedLocationKeepsItsOwner() {
        let id=UUID().uuidString,path="/private/flows/A/Main.kwipflow"
        let managed=ProjectRecord(id:id,name:"Current title",path:path,updatedAt:Date(timeIntervalSince1970:10))
        let external=ExternalFlowCatalog.Record(id:UUID(),projectID:id,title:"Old title",folderPath:"/private/flows/A",rootFilename:"Main.kwipflow",bookmarkData:Data([1]))
        let rows=RecentProjectCatalog.entries(managed:[managed,managed],external:[external,external])
        XCTAssertEqual(rows.count,1)
        XCTAssertEqual(rows.first?.source,.managed(managed))
    }
    func testIdenticalProjectIDsAndTitlesAtDifferentLocationsRemainSeparate() {
        let id=UUID().uuidString
        let a=ProjectRecord(id:id,name:"Model",path:"/private/A.kwipml",updatedAt:.distantPast)
        let b=ProjectRecord(id:id,name:"Model",path:"/private/B.kwipml",updatedAt:.distantPast)
        XCTAssertEqual(Set(RecentProjectCatalog.entries(managed:[a,b],external:[]).map(\.id)).count,2)
    }
    func testActualOpeningOrderWinsOverSaveDatesAndIsStableAcrossReload() throws {
        let a=ProjectRecord(id:UUID().uuidString,name:"Saved recently",path:"/private/A.kwipml",updatedAt:Date(timeIntervalSince1970:100))
        let b=ProjectRecord(id:UUID().uuidString,name:"Opened recently",path:"/private/B.kwipml",updatedAt:Date(timeIntervalSince1970:1))
        let folder=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:folder)}
        let history=RecentProjectHistory(directory:folder)
        try history.record(URL(fileURLWithPath:b.path),at:Date(timeIntervalSince1970:200))
        let restored=try RecentProjectHistory(directory:folder).load()
        XCTAssertEqual(RecentProjectCatalog.entries(managed:[a,b],external:[],opened:restored).first?.title,b.name)
    }
    func testUnbookmarkedPracticeCopiesDoNotLeakIntoOrdinaryRecentProjects()throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let workspace=WorkspaceStore(rootOverride:root,storageOverride:directory,detectCapabilities:false)
        let older=ProjectRecord(id:UUID().uuidString,name:"Same practice",path:"/private/old.kwipml",updatedAt:.distantPast)
        let newer=ProjectRecord(id:UUID().uuidString,name:"Same practice",path:"/private/new.kwipml",updatedAt:Date())
        XCTAssertTrue(workspace.lessonReading.associate(entryID:"output",projectID:older.id))
        XCTAssertTrue(workspace.lessonReading.saved(entryID:"output",projectID:newer.id))
        workspace.projects=[older,newer]
        XCTAssertTrue(workspace.modelProjects.isEmpty)
        XCTAssertEqual(workspace.lessonProjects.map(\.id),[newer.id],"Only the bookmark is listed as saved; old working copies remain on disk")
    }

}

extension RecentProjectCatalogTests {
    func testSameNamedCopiesHaveOnePresentationGroupButKeepEachLocation() {
        let a=ProjectRecord(id:UUID().uuidString,name:"Parallel classifier",path:"/private/A.kwipml",updatedAt:Date())
        let b=ProjectRecord(id:UUID().uuidString,name:"Parallel classifier",path:"/private/B.kwipml",updatedAt:.distantPast)
        let groups=RecentProjectGroup.make(RecentProjectCatalog.entries(managed:[a,b],external:[]))
        XCTAssertEqual(groups.count,1)
        XCTAssertEqual(groups.first?.entries.count,2)
        XCTAssertEqual(groups.first?.entries.first?.url.path,a.path)
    }
}
