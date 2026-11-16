import Foundation
import Darwin
import KwipMLCore

/// An immutable inventory capability. A cell contains only a hash, never a path;
/// opening it rechecks the captured regular file and its typed container.
struct FileImageArtifactResolver {
    let directory: URL
    private let directoryIdentity: WorkflowFileIdentity?
    private let files: [String: WorkflowArchiveFile]
    private let mediaFiles:[String:WorkflowArchiveFile]
    init(directory: URL, verifiedArtifacts: [WorkflowArchiveFile]) throws {
        var admitted: [String: WorkflowArchiveFile] = [:]
        for file in verifiedArtifacts where file.path.hasPrefix("image-") {
            guard WorkflowArchiveCodec.isHash(file.sha256), file.path == "image-\(file.sha256).kwipimg",
                  (1...50_000_000).contains(file.bytes), admitted[file.sha256] == nil else {
                throw WorkflowArchiveError("Captured image inventory is invalid.")
            }
            admitted[file.sha256] = file
        }
        var media:[String:WorkflowArchiveFile]=[:]
        for file in verifiedArtifacts where file.path.hasPrefix("media-") {
            guard WorkflowArchiveCodec.isHash(file.sha256),file.path=="media-"+file.sha256+".kwipmedia",(1...50_000_000).contains(file.bytes),media[file.sha256]==nil else {throw WorkflowArchiveError("Captured media inventory is invalid.")}
            media[file.sha256]=file
        }
        self.directory = directory; files = admitted;mediaFiles=media
        if admitted.isEmpty && media.isEmpty { directoryIdentity = nil }
        else {
            let fd = try WorkflowEvaluationOwnership.openDirectory(directory); defer { Darwin.close(fd) }
            directoryIdentity = try WorkflowEvaluationOwnership.directoryIdentity(fd)
        }
    }
    func contains(_ reference: FileImageReference) -> Bool { files[reference.sha256] != nil }
    func read(_ reference: FileImageReference) throws -> FileImageReference.Decoded {
        try Task.checkCancellation()
        guard let file = files[reference.sha256] else { throw WorkflowArchiveError("This image has no verified captured artifact.") }
        try verifyDirectory()
        let data = try DiagnosticFileIO.read(root: directory, parts: [file.path], limit: file.bytes)
        guard data.count == file.bytes, WorkflowArchiveCodec.hash(data) == file.sha256 else {
            throw WorkflowArchiveError("Captured image bytes differ from their recorded hash.")
        }
        try verifyDirectory()
        try Task.checkCancellation()
        return try FileImageReference.decode(container: data, expected: reference)
    }
    func contains(_ reference:MediaArtifactReference)->Bool {mediaFiles[reference.sha256] != nil}
    func read(_ reference:MediaArtifactReference)throws->MediaArtifactReference.Decoded {
        let bytes=try readMediaPin(reference.sha256)
        return try MediaArtifactReference.decode(container:bytes,expected:reference)
    }
    private func readMediaPin(_ pin:String)throws->Data {
        try Task.checkCancellation();guard let file=mediaFiles[pin] else {throw WorkflowArchiveError("This media has no verified captured artifact.")}
        try verifyDirectory();let bytes=try DiagnosticFileIO.read(root:directory,parts:[file.path],limit:file.bytes)
        guard bytes.count==file.bytes,WorkflowArchiveCodec.hash(bytes)==pin else {throw WorkflowArchiveError("Captured media bytes changed.")}
        try verifyDirectory();return bytes
    }
    func parentImage(_ reference:MediaArtifactReference)throws->FileImageReference? {
        guard case .string(let pin)?=reference.metadata["parentSHA256"],let file=files[pin] else {return nil}
        try verifyDirectory();let bytes=try DiagnosticFileIO.read(root:directory,parts:[file.path],limit:file.bytes)
        guard WorkflowArchiveCodec.hash(bytes)==pin,bytes.count==file.bytes else {throw WorkflowArchiveError("Captured parent image changed.")}
        let magic=Data("KWIPIMG1\n".utf8),start=13
        guard bytes.starts(with:magic),bytes.count>start else {throw WorkflowArchiveError("Invalid parent image container.")}
        let length=bytes[magic.count..<start].reduce(0){($0<<8)|Int($1)}
        guard (1...4096).contains(length),start+length<=bytes.count,case .object(let header)=try JSONDecoder().decode(JSONValue.self,from:bytes.subdata(in:start..<start+length)) else {throw WorkflowArchiveError("Invalid parent image header.")}
        var ref=header.filter {["schemaVersion","width","height","channels","dtype","layout"].contains($0.key)}
        ref["sha256"] = .string(pin);ref["kind"] = .string("imagePixelsRef")
        let expected=try FileImageReference.decode(JSONEncoder().encode(JSONValue.object(ref)))
        _=try FileImageReference.decode(container:bytes,expected:expected);try verifyDirectory();return expected
    }
    func parent(_ reference:MediaArtifactReference)throws->MediaArtifactReference? {
        guard case .string(let pin)?=reference.metadata["parentSHA256"],mediaFiles[pin] != nil else {return nil}
        let bytes=try readMediaPin(pin);let magic=Data("KWIPMEDIA1\n".utf8);let start=magic.count+4
        guard bytes.starts(with:magic),bytes.count>start else {throw WorkflowArchiveError("Unsupported media container.")}
        let length=bytes[magic.count..<start].reduce(0){($0<<8)|Int($1)}
        guard (1...8192).contains(length),start+length<=bytes.count,
              case .object(var fields)=try JSONDecoder().decode(JSONValue.self,from:bytes.subdata(in:start..<start+length)),case .string(let kind)?=fields["kind"] else {throw WorkflowArchiveError("Invalid parent media header.")}
        fields.removeValue(forKey:"byteOrder");fields["sha256"] = .string(pin);fields["kind"] = .string(kind == "audioSamples" ? "audioSamplesRef":"preparedTensorRef")
        let parent=try MediaArtifactReference.decode(JSONEncoder().encode(JSONValue.object(fields)))
        _=try MediaArtifactReference.decode(container:bytes,expected:parent);return parent
    }
    private func verifyDirectory() throws {
        let fd = try WorkflowEvaluationOwnership.openDirectory(directory); defer { Darwin.close(fd) }
        guard let directoryIdentity, try WorkflowEvaluationOwnership.directoryIdentity(fd) == directoryIdentity else {
            throw WorkflowArchiveError("Captured image directory changed after inventory verification.")
        }
    }
}

struct FileImageSelection: Identifiable {
    let reference: FileImageReference
    var id: String { reference.sha256 }
}

struct MediaArtifactSelection:Identifiable {let reference:MediaArtifactReference;var interval:ClosedRange<Double>? = nil;var id:String {reference.sha256+":"+String(interval?.lowerBound ?? -1)+":"+String(interval?.upperBound ?? -1)}}
