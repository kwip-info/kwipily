import Foundation
import KwipMLCore

/// Media decoding never grants access to original files or metadata paths.
enum MediaArtifactInventory {
    static func verify(entries:[JSONValue],files:[WorkflowArchiveFile],prefix:String,directory:URL,parts:[String])throws->[String:MediaArtifactReference] {
        var assets:[String:MediaArtifactReference]=[:]
        let pins=Set(entries.compactMap {entry->String? in guard case .object(let fields)=entry,case .string(let pin)?=fields["sha256"],[JSONValue.string("imagePixels"),.string("audioSamples"),.string("preparedTensor")].contains(fields["kind"] ?? .null) else {return nil};return pin})
        for case .object(let entry) in entries {
            guard [JSONValue.string("audioSamples"),.string("preparedTensor")].contains(entry["kind"] ?? .null) else {continue}
            try Task.checkCancellation()
            guard let media=entry["media"] else {throw failure("Media asset lacks its typed reference.")}
            let reference=try MediaArtifactReference.decode(JSONEncoder().encode(media))
            guard entry["kind"] == .string(reference.artifactKind),entry["path"] == .string(reference.filename),entry["sha256"] == .string(reference.sha256),
                  case .integer(let count)?=entry["bytes"],(1...Int64(MediaArtifactReference.maximumArtifactBytes)).contains(count),assets[reference.sha256]==nil,
                  files.contains(.init(path:prefix+reference.filename,bytes:Int(count),sha256:reference.sha256)) else {throw failure("Media asset differs from its owned inventory.")}
            if case .string(let parent)?=reference.metadata["parentSHA256"],!pins.contains(parent) {throw failure("Prepared media parent is missing from retained inventory.")}
            let bytes=try DiagnosticFileIO.read(root:directory,parts:parts+[reference.filename],limit:MediaArtifactReference.maximumArtifactBytes)
            guard bytes.count==count else {throw failure("Media asset size changed.")}
            _=try MediaArtifactReference.decode(container:bytes,expected:reference);assets[reference.sha256]=reference
        }
        return assets
    }
    static func references(_ value:JSONValue,assets:[String:MediaArtifactReference])throws {
        switch value {
        case .object(let fields):
            if [JSONValue.string("audioSamplesRef"),.string("preparedTensorRef")].contains(fields["kind"] ?? .null) {
                let reference=try MediaArtifactReference.decode(JSONEncoder().encode(value))
                guard assets[reference.sha256]==reference else {throw failure("Media reference lacks its immutable artifact.")}
            } else {for child in fields.values {try references(child,assets:assets)}}
        case .array(let children):for child in children {try references(child,assets:assets)}
        default:break
        }
    }
    private static func failure(_ text:String)->WorkflowArchiveError {.init(text)}
}
