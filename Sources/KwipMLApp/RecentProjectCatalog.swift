import Foundation
import KwipMLCore

/// Navigation metadata only. Listing never opens external documents or grants.
struct RecentProjectEntry: Identifiable, Equatable {
    enum Source: Equatable { case managed(ProjectRecord), external(ExternalFlowCatalog.Record) }
    let source: Source
    let openedAt: Date
    var url: URL { switch source {case .managed(let r): URL(fileURLWithPath:r.path);case .external(let r):r.rootURL} }
    var id: String {url.standardizedFileURL.path}
    var title: String {switch source {case .managed(let r):r.name;case .external(let r):r.title.isEmpty ? r.rootFilename:r.title}}
    var detail:String {
        let date=openedAt == .distantPast ? "Linked flow":openedAt.formatted(.dateTime.month().day().hour().minute())
        let location=url.lastPathComponent == "Main.kwipflow" ? url.deletingLastPathComponent().lastPathComponent:url.deletingPathExtension().lastPathComponent
        return date+" · "+(UUID(uuidString:location) == nil ? location:String(location.prefix(8)))
    }
    var isExternal: Bool {if case .external=source{return true};return false}
}

@MainActor enum RecentProjectCatalog {
    static func entries(managed:[ProjectRecord],external:[ExternalFlowCatalog.Record],opened:[String:Date]=[:])->[RecentProjectEntry] {
        var locations:[String:RecentProjectEntry]=[:]
        for record in managed.sorted(by:{$0.updatedAt>$1.updatedAt}) {
            let key=URL(fileURLWithPath:record.path).standardizedFileURL.path
            if locations[key]==nil {locations[key] = .init(source:.managed(record),openedAt:opened[key] ?? record.updatedAt)}
        }
        for record in external.sorted(by:{$0.id.uuidString<$1.id.uuidString}) {
            let key=record.rootURL.standardizedFileURL.path
            if locations[key]==nil {locations[key] = .init(source:.external(record),openedAt:opened[key] ?? .distantPast)}
        }
        return locations.values.sorted {$0.openedAt == $1.openedAt ? $0.id<$1.id:$0.openedAt>$1.openedAt}
    }
}

struct RecentProjectHistory {
    let directory:URL
    private struct Envelope:Codable {var schemaVersion=1;var opened:[String:Date]}
    func load()throws->[String:Date] {
        guard FileManager.default.fileExists(atPath:directory.appendingPathComponent("recent-projects.json").path) else{return [:]}
        let value=try JSONDecoder().decode(Envelope.self,from:DiagnosticFileIO.read(root:directory,parts:["recent-projects.json"],limit:1_000_000))
        guard value.schemaVersion==1,value.opened.count<=1000,value.opened.allSatisfy({$0.key.hasPrefix("/") && $0.key.utf8.count<=4096 && !$0.key.contains("\0") && $0.value.timeIntervalSince1970.isFinite}) else {throw WorkflowArchiveError("Recent-project metadata is invalid. Project files were retained.")}
        return value.opened
    }
    func record(_ url:URL,at date:Date=Date())throws {
        var opened=try load();opened[url.standardizedFileURL.path]=date
        let kept=opened.sorted{$0.value == $1.value ? $0.key<$1.key:$0.value>$1.value}.prefix(1000)
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        try DiagnosticFileIO.writeAtomic(JSONEncoder().encode(Envelope(opened:Dictionary(uniqueKeysWithValues:kept.map{($0.key,$0.value)}))),root:directory,parts:["recent-projects.json"])
    }
}

struct RecentProjectGroup:Identifiable {
    let title:String
    let entries:[RecentProjectEntry]
    var id:String {title}
    /// Presentation grouping only: every distinct location remains individually openable.
    static func make(_ entries:[RecentProjectEntry])->[Self] {
        var order:[String]=[],groups:[String:[RecentProjectEntry]]=[:]
        for entry in entries {if groups[entry.title]==nil {order.append(entry.title)};groups[entry.title,default:[]].append(entry)}
        return order.map{.init(title:$0,entries:groups[$0] ?? [])}
    }
}
