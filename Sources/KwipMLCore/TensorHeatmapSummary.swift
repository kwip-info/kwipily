import Foundation

/// Optional, versioned derived evidence. Values are aggregate measurements over
/// declared source ranges; this never promotes scalar samples to full coverage.
public struct TensorHeatmapSummary:Codable,Equatable {
    public struct Plane:Codable,Equatable {
        public let index:Int
        public let fixedIndices:[Int]
        public let mean:[Double?],minimum:[Double?],maximum:[Double?]
        public let finiteCount:[Int],nanCount:[Int],positiveInfinityCount:[Int],negativeInfinityCount:[Int]
    }
    public let schemaVersion:Int
    public let sourceShape:[Int]
    public let rowAxis:Int?,columnAxis:Int?
    public let aggregation:String?
    public let planes:[Plane]
    public let planeShape:[Int]?,gridShape:[Int]?,totalPlanes:Int?
    public let coverage:String
    public let reason:String?
    public var available:Bool {coverage=="complete" || coverage=="partialPlanes"}
    public init(value:JSONValue,expectedShape:[Int]?=nil)throws {
        guard case .object(let raw)=value,Set(raw.keys).isSubset(of:["schemaVersion","sourceShape","rowAxis","columnAxis","aggregation","planes","planeShape","gridShape","totalPlanes","coverage","reason"]) else {throw GraphConfigurationError("Unknown tensor heatmap fields.")}
        if case .array(let planes)?=raw["planes"] {
            let keys:Set<String>=["index","fixedIndices","mean","minimum","maximum","finiteCount","nanCount","positiveInfinityCount","negativeInfinityCount"]
            for plane in planes {guard case .object(let fields)=plane,Set(fields.keys)==keys else {throw GraphConfigurationError("Unknown or missing tensor heatmap plane fields.")}}
        }
        self=try JSONDecoder().decode(Self.self,from:JSONEncoder().encode(value))
        try validate(expectedShape:expectedShape)
    }
    public func validate(expectedShape:[Int]?=nil)throws {
        func require(_ condition:Bool)throws {if !condition {throw GraphConfigurationError("Invalid tensor heatmap coverage or tile measurements.")}}
        try require(schemaVersion==1 && sourceShape.count<=8 && sourceShape.allSatisfy {$0>=0 && $0<=100_000_000})
        if let expectedShape {try require(sourceShape==expectedShape)}
        try require(["complete","partialPlanes","empty","unavailable"].contains(coverage))
        if coverage=="unavailable" {try require(planes.isEmpty && !(reason ?? "").isEmpty && (reason?.count ?? 0)<=200);return}
        let height=sourceShape.count>=2 ? sourceShape[sourceShape.count-2]:1
        let width=sourceShape.last ?? 1
        var total=1
        for size in sourceShape.dropLast(min(2,sourceShape.count)) {
            let next=total.multipliedReportingOverflow(by:size);try require(!next.overflow);total=next.partialValue
        }
        try require(planeShape==[height,width] && totalPlanes==total && rowAxis==(sourceShape.count>=2 ? sourceShape.count-2:nil) && columnAxis==(sourceShape.isEmpty ? nil:sourceShape.count-1) && aggregation=="finiteMean")
        if coverage=="empty" {try require(sourceShape.contains(0) && planes.isEmpty && gridShape==[0,0]);return}
        guard let grid=gridShape,grid.count==2 else {throw GraphConfigurationError("Missing heatmap tile dimensions.")}
        try require(height>0 && width>0 && grid[0]>0 && grid[1]>0 && grid[0]<=min(height,64) && grid[1]<=min(width,64))
        let cells=grid[0]*grid[1]
        try require(!planes.isEmpty && planes.count<=32 && cells*planes.count<=8192 && planes.count<=total)
        try require(coverage=="complete" ? planes.count==total:planes.count<total)
        for (index,plane) in planes.enumerated() {
            try require(plane.index==index && plane.mean.count==cells && plane.minimum.count==cells && plane.maximum.count==cells)
            try require([plane.finiteCount,plane.nanCount,plane.positiveInfinityCount,plane.negativeInfinityCount].allSatisfy {$0.count==cells})
            var remaining=index;var fixed=Array(repeating:0,count:max(0,sourceShape.count-2))
            for axis in fixed.indices.reversed() {fixed[axis]=remaining%sourceShape[axis];remaining/=sourceShape[axis]}
            try require(plane.fixedIndices==fixed)
            for cell in 0..<cells {
                let r=cell/grid[1],c=cell%grid[1]
                let h=((r+1)*height+grid[0]-1)/grid[0]-(r*height+grid[0]-1)/grid[0]
                let w=((c+1)*width+grid[1]-1)/grid[1]-(c*width+grid[1]-1)/grid[1]
                let counts=[plane.finiteCount[cell],plane.nanCount[cell],plane.positiveInfinityCount[cell],plane.negativeInfinityCount[cell]]
                let population=h*w
                try require(counts.allSatisfy {$0>=0 && $0<=population} && counts.reduce(0,+)==population)
                if counts[0]==0 {try require(plane.mean[cell]==nil && plane.minimum[cell]==nil && plane.maximum[cell]==nil)}
                else {
                    guard let low=plane.minimum[cell],let high=plane.maximum[cell] else {throw GraphConfigurationError("Missing finite tile extrema.")}
                    try require(low.isFinite && high.isFinite && low<=high)
                    if let mean=plane.mean[cell] {try require(mean.isFinite && mean>=low-max(abs(low)*1e-12,1e-12) && mean<=high+max(abs(high)*1e-12,1e-12))}
                }
            }
        }
    }
}
