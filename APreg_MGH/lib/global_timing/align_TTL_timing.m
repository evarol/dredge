function [slave_timestamps] = align_TTL_timing(master_TTL_data, master_timestamps, slave_TTL_data)
    
    master_tresh = median(unique(abs(diff(master_timestamps))));
    slave_tresh = median(unique(abs(diff(slave_TTL_data))));

    master_TTL = find(diff(master_TTL_data)>master_tresh);
    slave_TTL = find(diff(slave_TTL_data)>slave_tresh);


    if isempty(master_timestamps)
        master_timestamps = [1:master_TTL(end)];
    end
    
%     stem(master_TTL,ones(size(master_TTL)))
%     hold on
%     stem(slave_TTL,0.7*ones(size(slave_TTL)))

    % TO align: shift to match first TTL : 
    
    scale = (master_TTL(end)-master_TTL(1)) /  (slave_TTL(end)-slave_TTL(1));
    
    shift =  master_TTL(1)- slave_TTL(1)*scale;

    master_timebins = [1:length(master_timestamps)];
    
    slave_timebins = (master_timebins*scale+shift);
    
    slave_timebins = slave_timebins(1:min(length(slave_timebins), length(slave_TTL_data)));
    
    slave_timestamps = interp1(master_timebins,master_timestamps,slave_timebins);

    slave_timestamps = slave_timestamps(1:min(length(slave_timestamps),length(slave_TTL_data)));
    
    if length(slave_timestamps) < length(slave_TTL_data)
        slave_timestamps = interp1(1:length(slave_timestamps),slave_timestamps,1:length(slave_TTL_data),'linear', 'extrap');
    end
    
    
    slave_TTL_data = slave_TTL_data(1:length(slave_timestamps));
        
    
    figure;
    subplot(2,2,[1,2]);
    plot(master_timestamps,master_TTL_data./max(master_TTL_data))
    hold on
    plot(slave_timestamps,0.7*slave_TTL_data(1:length(slave_timestamps))./max(slave_TTL_data))
    title('Verfiy alignment:')
    legend('master','slave')
    xlabel('time (sec)')
    
    subplot(2,2,[3]);
    plot(master_timestamps,master_TTL_data./max(master_TTL_data))
    hold on
    plot(slave_timestamps,0.7*slave_TTL_data(1:length(slave_timestamps))./max(slave_TTL_data))
    title('Verfiy alignment:')
    legend('master','slave')
    xlabel('time (sec)')
    xlim([master_timestamps(1),master_timestamps(end)*0.05])
    
    subplot(2,2,[4]);
    plot(master_timestamps,master_TTL_data./max(master_TTL_data))
    hold on
    plot(slave_timestamps,0.7*slave_TTL_data(1:length(slave_timestamps))./max(slave_TTL_data))
    title('Verfiy alignment:')
    legend('master','slave')
    xlabel('time (sec)')
    xlim([master_timestamps(end)*0.95,master_timestamps(end)])
    