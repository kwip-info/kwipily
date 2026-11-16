import Foundation
import KwipMLCore

/// Pins pixel assets before native presentation. Cells never resolve arbitrary paths.
enum FileImageArtifactInventory {
    @discardableResult
    static func verify(manifest:Data,files:[WorkflowArchiveFile],prefix:String,directory:URL,readPrefix:[String]?=nil)throws->[String:FileImageReference] {
        guard manifest.count<=2_000_000,case .object(let root)=try JSONDecoder().decode(JSONValue.self,from:manifest),case .array(let entries)?=root["artifacts"] else {throw failure("Missing image artifact inventory.")}
        let parts=readPrefix ?? prefix.split(separator:"/").map(String.init)
        let mediaAssets=try MediaArtifactInventory.verify(entries:entries,files:files,prefix:prefix,directory:directory,parts:parts)
        var assets:[String:FileImageReference]=[:]
        var previews:[String:JSONValue]?
        for case .object(let entry) in entries {
            if entry["path"] == .string("previews.json"),entry["kind"] == .string("previews") {previews=entry}
            guard entry["kind"] == .string("imagePixels") else {continue}
            try Task.checkCancellation()
            guard let image=entry["image"] else {throw failure("Image asset lacks its typed reference.")}
            let reference=try FileImageReference.decode(JSONEncoder().encode(image))
            let name="image-"+reference.sha256+".kwipimg"
            guard entry["path"] == .string(name),entry["sha256"] == .string(reference.sha256),
                  case .integer(let count)?=entry["bytes"],(1...Int64(FileImageReference.maximumArtifactBytes)).contains(count),
                  assets[reference.sha256]==nil,
                  files.contains(.init(path:prefix+name,bytes:Int(count),sha256:reference.sha256)) else {throw failure("Image asset differs from its owned inventory.")}
            let bytes=try DiagnosticFileIO.read(root:directory,parts:parts+[name],limit:FileImageReference.maximumArtifactBytes)
            guard bytes.count==count else {throw failure("Image asset size changed.")}
            _=try FileImageReference.decode(container:bytes,expected:reference)
            assets[reference.sha256]=reference
        }
        if let entry=previews {
            guard case .integer(let count)?=entry["bytes"],(0...32_000_000).contains(count),case .string(let sha)?=entry["sha256"],
                  files.contains(.init(path:prefix+"previews.json",bytes:Int(count),sha256:sha)) else {throw failure("Image preview lacks its owned inventory pin.")}
            let bytes=try DiagnosticFileIO.read(root:directory,parts:parts+["previews.json"],limit:32_000_000)
            guard bytes.count==count,WorkflowArchiveCodec.hash(bytes)==sha else {throw failure("Image preview differs from its immutable pin.")}
            let preview=try JSONDecoder().decode(JSONValue.self,from:bytes)
            try references(preview,assets:assets)
            try MediaArtifactInventory.references(preview,assets:mediaAssets)
        }
        return assets
    }
    private static func references(_ value:JSONValue,assets:[String:FileImageReference])throws {
        switch value {
        case .object(let fields):
            if fields["kind"] == .string("imagePixelsRef") {
                let reference=try FileImageReference.decode(JSONEncoder().encode(value))
                guard assets[reference.sha256]==reference else {throw failure("Image preview reference is missing from its immutable inventory.")}
            } else {for child in fields.values {try references(child,assets:assets)}}
        case .array(let values):for child in values {try references(child,assets:assets)}
        default:break
        }
    }
    private static func failure(_ message:String)->WorkflowArchiveError {.init(message)}
}
