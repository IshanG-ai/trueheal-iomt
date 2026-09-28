#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/csma-module.h"
#include "ns3/applications-module.h"
using namespace ns3;

int main(int argc, char *argv[]) {
  std::string scenario = "benign";
  std::string pcapDir = "ns3_pcaps";
  double duration = 15.0;
  CommandLine cmd;
  cmd.AddValue("scenario", "scenario tag", scenario);
  cmd.AddValue("pcapDir", "output dir", pcapDir);
  cmd.AddValue("duration", "simulation seconds", duration);
  cmd.Parse(argc, argv);

  NodeContainer nodes;
  nodes.Create(6);   // 0-3 sensors, 4 gateway, 5 attacker
  CsmaHelper csma;
  csma.SetChannelAttribute("DataRate", StringValue("100Mbps"));
  csma.SetChannelAttribute("Delay", TimeValue(MilliSeconds(2)));
  NetDeviceContainer devs = csma.Install(nodes);

  InternetStackHelper inet;
  inet.Install(nodes);
  Ipv4AddressHelper addr;
  addr.SetBase("10.0.0.0", "255.255.255.0");
  Ipv4InterfaceContainer ifaces = addr.Assign(devs);

  uint16_t port = 9;
  uint64_t rate = 5000;      // bit/s
  uint32_t pktSize = 512;

  if (scenario == "dos_tcp" || scenario == "ddos_tcp")   { rate = 500000; pktSize = 1024; }
  if (scenario == "dos_udp" || scenario == "ddos_udp")   { rate = 600000; pktSize = 1024; }
  if (scenario == "dos_syn" || scenario == "ddos_syn")   { rate = 400000; pktSize = 64; }
  if (scenario == "dos_icmp" || scenario == "ddos_icmp") { rate = 450000; pktSize = 64; }
  if (scenario.find("mqtt") != std::string::npos)        { port = 1883; rate = 100000; pktSize = 128; }
  if (scenario == "arp_spoofing")                        { rate = 80000; pktSize = 28; }
  if (scenario == "recon_portscan" || scenario == "recon_osscan") { rate = 30000; pktSize = 64; }
  if (scenario == "recon_pingsweep")                     { rate = 20000; pktSize = 64; }
  if (scenario == "recon_vulnscan")                      { rate = 15000; pktSize = 64; }

  bool udp = scenario.find("udp") != std::string::npos;
  std::string atkFactory = udp ? "ns3::UdpSocketFactory" : "ns3::TcpSocketFactory";

  // Normal sensor traffic (TCP): nodes 0-3 -> gateway
  for (int i = 0; i < 4; i++) {
    OnOffHelper src("ns3::TcpSocketFactory", InetSocketAddress(ifaces.GetAddress(4), port));
    src.SetConstantRate(DataRate(5000UL), 512);
    ApplicationContainer a = src.Install(nodes.Get(i));
    a.Start(Seconds(0.1));
    a.Stop(Seconds(duration));
  }

  // Attack traffic: node 5 -> gateway (4 parallel sources for DDoS scenarios)
  if (scenario != "benign") {
    uint32_t nAtk = (scenario.find("ddos") != std::string::npos) ? 4 : 1;
    for (uint32_t k = 0; k < nAtk; k++) {
      OnOffHelper atk(atkFactory, InetSocketAddress(ifaces.GetAddress(4), port));
      atk.SetConstantRate(DataRate(rate), pktSize);
      ApplicationContainer a = atk.Install(nodes.Get(5));
      a.Start(Seconds(0.5));
      a.Stop(Seconds(duration));
    }
  }

  // Sinks (TCP + UDP) on gateway
  PacketSinkHelper tcpSink("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
  tcpSink.Install(nodes.Get(4)).Start(Seconds(0.0));
  PacketSinkHelper udpSink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
  udpSink.Install(nodes.Get(4)).Start(Seconds(0.0));

  csma.EnablePcap(pcapDir + "/" + scenario, devs.Get(4), true);   // capture at gateway

  Simulator::Stop(Seconds(duration));
  Simulator::Run();
  Simulator::Destroy();
  return 0;
}
